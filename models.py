"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import os.path as osp

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import MotEncoderTra
from position_encoding import PositionEmbeddingSine

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            # self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm1 = nn.GroupNorm(8, dim_in, affine=True)
            self.norm2 = nn.GroupNorm(8, dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class LmkEncoder(nn.Module):
    def __init__(self):
        super(LmkEncoder, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(68*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )
        
    def forward(self, input_lmk):
        # print('example_landmark', input_lmk.shape) # batch, 68, 3
        b, l, k, d= input_lmk.shape
        example_landmark = input_lmk[:,:,:,:2].reshape(b*l, k, 2).reshape(b*l, k*2) # only consider two dim for now
        example_landmark_f = self.lmark_encoder(example_landmark).reshape(b, l, 512)
        # print(example_landmark_f.shape) # batch, 512
        return example_landmark_f

class LmkEncoderPCA(nn.Module):
    def __init__(self):
        super(LmkEncoderPCA, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(936,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )
        
    def forward(self, input_lmk):
        # print('example_landmark', input_lmk.shape) # batch, 468, 3
        b, k, d= input_lmk.shape
        example_landmark = input_lmk[:,:,:2].reshape(b, k*2) # only consider two dim for now
        example_landmark_f = self.lmark_encoder(example_landmark).reshape(b, 512)
        # print(example_landmark_f.shape) # batch, 512
        return example_landmark_f

# LSTM of Mot Decoder
class MotDecoder(nn.Module):
    def __init__(self):
        super(MotDecoder, self).__init__()
        # audio to landmark
        self.extract_feature = nn.Sequential(
            nn.Linear(5632,2048),
            nn.ReLU(True),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            )

        self.lstm = nn.LSTM(1024, 256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,68*2),#20
            )
    
    def forward(self, f_landmark, lstm_input, length_mots):
        # print(lstm_input.shape, f_landmark.shape) # b 64 80 length/ b 512
        b, d, n_mel, length = lstm_input.shape
        lstm_input = lstm_input.reshape(b, d*n_mel, length)
        # f_landmark = f_landmark.unsqueeze(-1).repeat(1, 1, length)
        f_landmark = f_landmark.permute(0, 2, 1)

        lstm_input = torch.cat([lstm_input, f_landmark], 1).permute(0,2,1).reshape(b*length, d*n_mel+512)
        # print(lstm_input.shape) # b, length, 5632
        lstm_input = self.extract_feature(lstm_input).reshape(b, length, 1024)
        # print(lstm_input.shape) # b, length, 1024

        hidden = (torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                  torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])

        fc_out = self.lstm_fc(lstm_out.reshape(b*length, 256)).reshape(b, length, 68*2)

  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]

        return fc_out

# LSTM of Mot Decoder PCA
class MotDecoderPCA(nn.Module):
    def __init__(self):
        super(MotDecoderPCA, self).__init__()
        # audio to landmark
        self.extract_feature = nn.Sequential(
            nn.Linear(5632,2048),
            nn.ReLU(True),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            )

        self.lstm = nn.LSTM(1024, 256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256, 32),#20
            )
    
    def forward(self, f_landmark, lstm_input, length_mots):
        # print(lstm_input.shape, f_landmark.shape) # b 64 80 length/ b 512
        b, d, n_mel, length = lstm_input.shape
        lstm_input = lstm_input.reshape(b, d*n_mel, length)
        f_landmark = f_landmark.unsqueeze(-1).repeat(1, 1, length)
        # f_landmark = f_landmark.permute(0, 2, 1)

        lstm_input = torch.cat([lstm_input, f_landmark], 1).permute(0,2,1).reshape(b*length, d*n_mel+512)
        # print(lstm_input.shape) # b, length, 5632
        lstm_input = self.extract_feature(lstm_input).reshape(b, length, 1024)
        # print(lstm_input.shape) # b, length, 1024

        hidden = (torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                  torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])


       # lstm_input = torch.stack(lstm_input, dim = 1) #connect torch.Size([16, 16, 768])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])

        fc_out = self.lstm_fc(lstm_out.reshape(b*length, 256)).reshape(b, length, 32)

  #      features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
  #      features = torch.unsqueeze(features,2)
  #      features = torch.unsqueeze(features,3)
  #      x = 90*self.decon(features) #[1, 1,28, 12]

        return fc_out

# LSTM of Mot Decoder PCA
class MotDecoderPCATR(nn.Module):
    def __init__(self):
        super(MotDecoderPCATR, self).__init__()
        # audio to landmark
        self.extract_feature = nn.Sequential(
            nn.Linear(5632,2048),
            nn.ReLU(True),
            nn.Linear(2048,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        # self.lstm = nn.LSTM(1024, 256,3,batch_first = True)
        self.trans_fc = nn.Sequential(
            nn.Linear(512, 32),#20
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1)
            
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.03)
                nn.init.xavier_uniform_(m.weight, gain=1)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # init()
    
    def forward(self, f_landmark, lstm_input, length_mots):
        # print(lstm_input.shape, f_landmark.shape) # b 64 80 length/ b 512
        b, d, n_mel, l = lstm_input.shape
        trans_input = lstm_input.reshape(b, d*n_mel, l)
        f_landmark = f_landmark.unsqueeze(-1).repeat(1, 1, l)
        # f_landmark = f_landmark.permute(0, 2, 1)

        trans_input = torch.cat([trans_input,f_landmark], 1).permute(0,2,1).reshape(b*l, d*n_mel+512)
        # print(trans_input.shape)
        trans_input = self.extract_feature(trans_input).reshape(b, l, 512).permute(0, 2, 1).unsqueeze(2)

        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        # print('trans mask: ', mask, mask.shape)
        # assert(0)
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        # print(pos_emb[0, 510, 0, :], pos_emb.shape)
        # print(pos_emb[0, 251, 0, :], pos_emb.shape)

        out = self.metrans(trans_input, mask, pos_emb)
        # print('transformer out: ', out, out.shape)
        fc_out = self.trans_fc(out.reshape(b*l, 512)).reshape(b, l, 32)

        return fc_out

class MotDecoderTR(nn.Module):
    def __init__(self):
        super(MotDecoderTR, self).__init__()
        # audio to landmark
        self.extract_feature = nn.Sequential(
            nn.Linear(5632,2048),
            nn.ReLU(True),
            nn.Linear(2048,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        # self.lstm = nn.LSTM(1024, 256,3,batch_first = True)
        self.trans_fc = nn.Sequential(
            nn.Linear(512,68*2),#20
            )
    
    def forward(self, f_landmark, trans_input, length_mots):
        b, d, n_mel, l = trans_input.shape
        trans_input = trans_input.reshape(b, d*n_mel, l)
        f_landmark = f_landmark.permute(0, 2, 1)
        trans_input = torch.cat([trans_input,f_landmark], 1).permute(0,2,1).reshape(b*l, d*n_mel+512)
        trans_input = self.extract_feature(trans_input).reshape(b, l, 512).permute(0, 2, 1).unsqueeze(2)

        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        # print('trans mask: ', mask, mask.shape)
        # assert(0)
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        # print(pos_emb[0, 510, 0, :], pos_emb.shape)
        # print(pos_emb[0, 251, 0, :], pos_emb.shape)

        out = self.metrans(trans_input, mask, pos_emb)
        # print('transformer out: ', out, out.shape)
        fc_out = self.trans_fc(out.reshape(b*l, 512)).reshape(b, l, 68*2)

        return fc_out

# class MotDecoderTRv2(nn.Module):
#     def __init__(self):
#         super(MotDecoderTR, self).__init__()
#         # audio to landmark
#         self.extract_feature = nn.Sequential(
#             nn.Linear(5632,2048),
#             nn.ReLU(True),
#             nn.Linear(2048,512),
#             nn.ReLU(True),
#             )

#         self.position_embedding = PositionEmbeddingSine(512, normalize=True)
#         self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
#                                      dim_feedforward=1024, dropout=0.1,
#                                      activation="relu", normalize_before=False)

#         # self.lstm = nn.LSTM(1024, 256,3,batch_first = True)
#         self.trans_fc = nn.Sequential(
#             nn.Linear(512,68*2),#20
#             )
    
#     def forward(self, f_landmark, trans_input, length_mots):
#         b, d, n_mel, l = trans_input.shape
#         trans_input = trans_input.reshape(b, d*n_mel, l)
        
#         # f_landmark = f_landmark.unsqueeze(-1).repeat(1, 1, l)
#         trans_input = torch.cat([trans_input,f_landmark], 1).permute(0,2,1).reshape(b*l, d*n_mel+512)
#         trans_input = self.extract_feature(trans_input).reshape(b, l, 512).permute(0, 2, 1).unsqueeze(2)

#         mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
#         for index, lm in enumerate(length_mots):
#             mask[index, :lm] = False
#         # print('trans mask: ', mask, mask.shape)
#         # assert(0)
#         pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
#         # print(pos_emb[0, 510, 0, :], pos_emb.shape)
#         # print(pos_emb[0, 251, 0, :], pos_emb.shape)

#         out = self.metrans(trans_input, mask, pos_emb)
#         # print('transformer out: ', out, out.shape)
#         fc_out = self.trans_fc(out.reshape(b*l, 512)).reshape(b, l, 68*2)

#         return fc_out

class Generator(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48*8, w_hpf=1, F0_channel=0, audio=False):
        super().__init__()
        self.audio = audio
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        # down/up-sampling blocks
        repeat_num = 4 #int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
        
        # F0 blocks 
        if F0_channel != 0:
            self.decode.insert(
                0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out, style_dim, w_hpf=w_hpf))
        
        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(
                    0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf))
        
        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2), normalize=True, downsample="half"),
            )
        

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, F0=None):            
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            
        if F0 is not None:
            F0 = self.F0_conv(F0)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])
            x = torch.cat([x, F0], axis=1)
#             print('model 230 x+F0 shape:', x.shape) # 5,74?

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        # print('model 303 generator output:', x.shape, self.to_out(x).shape) # model 237 generator output: torch.Size([b, 64(c), 80(numl), 296(length)]) torch.Size([1, 1, 80, 296])
        if self.audio:
            return self.to_out(x)
        else:
            return self.to_out(x), x


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=48, num_domains=2, hidden_dim=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        
        # real/fake discriminator
        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)                             
        self.num_domains = num_domains
        
    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

# LSTM version of DiscriminatorLmk
class DiscriminatorLmk(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(68*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )
        self.lstm = nn.LSTM(512,256,3,batch_first = True)
        # self.lstm_fc = nn.Sequential(
        #     nn.Linear(256, num_domains),
        #     nn.Tanh())
        self.decision = nn.Sequential(
            nn.Linear(256,1),
            )
        self.aggregator = nn.AdaptiveAvgPool1d(1)
        # self.activate = nn.Sigmoid()

    def get_feature(self, x):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        lstm_input = self.extract_feature(x)
        lstm_input = lstm_input.view(b, l, -1)  # (batch, length, dims)
        hidden = (torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                  torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])

        decision = self.decision(lstm_out.reshape(b*l, 256)).reshape(b, l)
        return decision

    def forward(self, x, length):
        decision = self.get_feature(x)
        ds = []
        for index, ll in enumerate(length):
            ds.append(self.aggregator(decision[index, :ll].reshape(1,1,ll)))
        decision = torch.cat(ds, 2).squeeze()
        return decision

class DiscriminatorLmkTR(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(68*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        # self.trans_fc = nn.Sequential(
        #     nn.Linear(256, num_domains),
        #     nn.Tanh())
        self.decision = nn.Sequential(
            nn.Linear(512,1),
            )
        self.aggregator = nn.AdaptiveAvgPool1d(1)
        # self.activate = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1)
            
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.03)
                nn.init.xavier_uniform_(m.weight, gain=1)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def get_feature(self, x, length_mots):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        trans_input = self.extract_feature(x)
        trans_input = trans_input.view(b, l, -1).permute(0, 2, 1).unsqueeze(2)

        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        out = self.metrans(trans_input, mask, pos_emb)

        decision = self.decision(out.reshape(b*l, 512)).reshape(b, l)
        return decision

    def forward(self, x, length_mots):
        decision = self.get_feature(x, length_mots)
        ds = []
        for index, ll in enumerate(length_mots):
            ds.append(self.aggregator(decision[index, :ll].reshape(1,1,ll)))
        decision = torch.cat(ds, 2).squeeze()
        return decision

class DiscriminatorLmkTR468(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(468*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        self.tran_fc = nn.Sequential(
            nn.Linear(512, num_domains),
            nn.Tanh())
        
        self.aggregator = nn.AdaptiveAvgPool1d(1)
        # self.activate = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1)
            
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.03)
                nn.init.xavier_uniform_(m.weight, gain=1)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def get_feature(self, x, length_mots):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        trans_input = self.extract_feature(x)
        trans_input = trans_input.view(b, l, -1).permute(0, 2, 1).unsqueeze(2)

        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        out = self.metrans(trans_input, mask, pos_emb)

        decision = self.tran_fc(out.reshape(b*l, 512)).reshape(b, l, -1)
        return decision

    def forward(self, x, y, length_mots):
        dec_out = self.get_feature(x, length_mots)
        # ds = []
        # for index, ll in enumerate(length_mots):
        #     ds.append(self.aggregator(decision[index, :ll].reshape(1,1,ll)))
        # decision = torch.cat(ds, 2).squeeze()
        outs = []
        for index, ll in enumerate(length_mots):
            tmp = dec_out[index, :ll, :].permute(1,0)
            outs.append(self.aggregator(tmp.reshape(1, self.num_domians, ll)))
        
        out = torch.cat(outs, 0).squeeze()
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

# LSTM version of EmoClassifier
class EmoClassifier(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(68*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )
        self.lstm = nn.LSTM(512,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256, num_domains),
            nn.Tanh())
        self.aggregator = nn.AdaptiveAvgPool1d(1)

    def get_feature(self, x):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        lstm_input = self.extract_feature(x)
        lstm_input = lstm_input.view(b, l, -1)  # (batch, length, dims)
        hidden = (torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                  torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])
        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        fc_out = self.lstm_fc(lstm_out.reshape(b*l, 256)).reshape(b, l, -1)

        return fc_out

    def forward(self, x, length):
        fc_out = self.get_feature(x)
        outs = []
        for index, ll in enumerate(length):
            tmp = fc_out[index, :ll, :].permute(1,0)
            outs.append(self.aggregator(tmp.reshape(1, self.num_domians, ll)))
        cls = torch.cat(outs, 0).squeeze()
        return cls

class EmoClassifierTR(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(68*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        self.tran_fc = nn.Sequential(
            nn.Linear(512, num_domains),
            nn.Tanh())
        
        self.aggregator = nn.AdaptiveAvgPool1d(1)

    def get_feature(self, x, length_mots):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        trans_input = self.extract_feature(x)
        trans_input = trans_input.view(b, l, -1).permute(0, 2, 1).unsqueeze(2)

        # transformer
        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        out = self.metrans(trans_input, mask, pos_emb)

        fc_out = self.tran_fc(out.reshape(b*l, 512)).reshape(b, l, -1)

        return fc_out

    def forward(self, x, length_mots):
        fc_out = self.get_feature(x, length_mots)
        outs = []
        for index, ll in enumerate(length_mots):
            tmp = fc_out[index, :ll, :].permute(1,0)
            outs.append(self.aggregator(tmp.reshape(1, self.num_domians, ll)))
        cls = torch.cat(outs, 0).squeeze()
        return cls

class EmoClassifierTR468(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_domians = num_domains
        self.extract_feature = nn.Sequential(
            nn.Linear(468*2,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            )

        self.position_embedding = PositionEmbeddingSine(512, normalize=True)
        self.metrans = MotEncoderTra(d_model=512, nhead=8, num_encoder_layers=3,
                                     dim_feedforward=1024, dropout=0.1,
                                     activation="relu", normalize_before=False)

        self.tran_fc = nn.Sequential(
            nn.Linear(512, num_domains),
            nn.Tanh())
        
        self.aggregator = nn.AdaptiveAvgPool1d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1)
            
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.03)
                nn.init.xavier_uniform_(m.weight, gain=1)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_feature(self, x, length_mots):
        b, l, m, d = x.shape
        x = x.reshape(b, l, m*d).reshape(b*l, m*d)
        trans_input = self.extract_feature(x)
        trans_input = trans_input.view(b, l, -1).permute(0, 2, 1).unsqueeze(2)

        # transformer
        mask = torch.ones((b, l), dtype=torch.bool).to(trans_input.device)
        for index, lm in enumerate(length_mots):
            mask[index, :lm] = False
        pos_emb = self.position_embedding(trans_input, mask.unsqueeze(1))
        out = self.metrans(trans_input, mask, pos_emb)

        fc_out = self.tran_fc(out.reshape(b*l, 512)).reshape(b, l, -1)

        return fc_out

    def forward(self, x, length_mots):
        fc_out = self.get_feature(x, length_mots)
        outs = []
        for index, ll in enumerate(length_mots):
            tmp = fc_out[index, :ll, :].permute(1,0)
            outs.append(self.aggregator(tmp.reshape(1, self.num_domians, ll)))
        cls = torch.cat(outs, 0).squeeze()
        return cls

def motion_to_lmk2d(initlm, fakemotion):
    #####
    ### change motion and initlm to fake landmark
    #####
    fl = []
    fl.append(initlm[:,:,:2])
    # calculate fake landmarks
    for i in range(fakemotion.shape[1]):
        fl.append(fl[i] + fakemotion[:,i,:,:2])
    fl = torch.stack(fl, dim=1)
    return fl

class DiscriminatorMot(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        
        # real/fake discriminator
        self.dis = DiscriminatorLmkTR(num_domains=num_domains)
        # adversarial classifier
        self.cls = EmoClassifierTR(num_domains=num_domains)
        self.num_domains = num_domains
        
    def forward(self, example_lmk, motion, length):
        # print(example_lmk.shape, motion.shape, length)

        lmk = motion_to_lmk2d(example_lmk, motion)
        return self.dis(lmk, length)

    def classifier(self, example_lmk, motion, length):
        lmk = motion_to_lmk2d(example_lmk, motion)
        return self.cls(lmk, length)

def pca2lmk(pca, U, mean):
    b, l, d = pca.shape
    pca = pca.reshape(b*l, d)
    lmk = torch.mm(pca, U.t())
    lmk = lmk + mean.expand_as(lmk)
    lmk = lmk.reshape(b, l, -1).reshape(b, l, 468, 2)
    return lmk

#PCA of discriminator
class DiscriminatorMotPCA(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        
        # real/fake discriminator
        self.dis = DiscriminatorLmkTR468(num_domains=num_domains)
        # adversarial classifier
        self.cls = EmoClassifierTR468(num_domains=num_domains)
        self.num_domains = num_domains

        self.mean_mead = torch.from_numpy(np.load('./PCA/mean_mead.npy').astype(np.float32)).cuda()
        self.U = torch.from_numpy(np.load('./PCA/U_mead.npy').astype(np.float32))[:,:32].cuda()

        
    def forward(self, example_lmk, y, fake_pca, length):
        # print(example_lmk.shape, motion.shape, length)

        lmk = pca2lmk(fake_pca, self.U, self.mean_mead)
        return self.dis(lmk, y, length)

    def classifier(self, example_lmk, fake_pca, length):
        lmk = pca2lmk(fake_pca, self.U, self.mean_mead)
        return self.cls(lmk, length)

def build_model_audio(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel, audio=True)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    device_ids = [0,1,2]

    generator = torch.nn.DataParallel(generator, device_ids=device_ids, output_device=0)
    mapping_network = torch.nn.DataParallel(mapping_network, device_ids=device_ids, output_device=0)
    style_encoder = torch.nn.DataParallel(style_encoder, device_ids=device_ids, output_device=0)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids, output_device=0)
    generator_ema = torch.nn.DataParallel(generator_ema, device_ids=device_ids, output_device=0)
    mapping_network_ema = torch.nn.DataParallel(mapping_network_ema, device_ids=device_ids, output_device=0)
    style_encoder_ema = torch.nn.DataParallel(style_encoder_ema, device_ids=device_ids, output_device=0)
    
    F0_model = torch.nn.DataParallel(F0_model, device_ids=device_ids, output_device=0)
    ASR_model = torch.nn.DataParallel(ASR_model, device_ids=device_ids, output_device=0)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema

def build_model(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    lmk_encoder = LmkEncoder()
    mot_decoder = MotDecoderTR()
    discriminator_mot = DiscriminatorMot(args.num_domains)

    # device_ids = [3]
    # output_device = 0

    # generator = torch.nn.DataParallel(generator, device_ids=device_ids, output_device=output_device)
    # mapping_network = torch.nn.DataParallel(mapping_network, device_ids=device_ids, output_device=output_device)
    # style_encoder = torch.nn.DataParallel(style_encoder, device_ids=device_ids, output_device=output_device)
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids, output_device=output_device)
    # generator_ema = torch.nn.DataParallel(generator_ema, device_ids=device_ids, output_device=output_device)
    # mapping_network_ema = torch.nn.DataParallel(mapping_network_ema, device_ids=device_ids, output_device=output_device)
    # style_encoder_ema = torch.nn.DataParallel(style_encoder_ema, device_ids=device_ids, output_device=output_device)
    
    # F0_model = torch.nn.DataParallel(F0_model, device_ids=device_ids, output_device=output_device)
    # ASR_model = torch.nn.DataParallel(ASR_model, device_ids=device_ids, output_device=output_device)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    nets_lmk = Munch(lmk_encoder=lmk_encoder,
                     mot_decoder=mot_decoder,
                     discriminator_mot=discriminator_mot)

    return nets, nets_ema, nets_lmk

def build_model_lstm(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    lmk_encoder = LmkEncoder()
    mot_decoder = MotDecoder()
    discriminator_mot = DiscriminatorMot(args.num_domains)

    # device_ids = [3]
    # output_device = 0

    # generator = torch.nn.DataParallel(generator, device_ids=device_ids, output_device=output_device)
    # mapping_network = torch.nn.DataParallel(mapping_network, device_ids=device_ids, output_device=output_device)
    # style_encoder = torch.nn.DataParallel(style_encoder, device_ids=device_ids, output_device=output_device)
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids, output_device=output_device)
    # generator_ema = torch.nn.DataParallel(generator_ema, device_ids=device_ids, output_device=output_device)
    # mapping_network_ema = torch.nn.DataParallel(mapping_network_ema, device_ids=device_ids, output_device=output_device)
    # style_encoder_ema = torch.nn.DataParallel(style_encoder_ema, device_ids=device_ids, output_device=output_device)
    
    # F0_model = torch.nn.DataParallel(F0_model, device_ids=device_ids, output_device=output_device)
    # ASR_model = torch.nn.DataParallel(ASR_model, device_ids=device_ids, output_device=output_device)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    nets_lmk = Munch(lmk_encoder=lmk_encoder,
                     mot_decoder=mot_decoder,
                     discriminator_mot=discriminator_mot)

    return nets, nets_ema, nets_lmk

def build_model_lstm_pca(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    lmk_encoder = LmkEncoderPCA()
    mot_decoder = MotDecoderPCA()
    discriminator_mot = DiscriminatorMotPCA(args.num_domains)

    # device_ids = [3]
    # output_device = 0

    # generator = torch.nn.DataParallel(generator, device_ids=device_ids, output_device=output_device)
    # mapping_network = torch.nn.DataParallel(mapping_network, device_ids=device_ids, output_device=output_device)
    # style_encoder = torch.nn.DataParallel(style_encoder, device_ids=device_ids, output_device=output_device)
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids, output_device=output_device)
    # generator_ema = torch.nn.DataParallel(generator_ema, device_ids=device_ids, output_device=output_device)
    # mapping_network_ema = torch.nn.DataParallel(mapping_network_ema, device_ids=device_ids, output_device=output_device)
    # style_encoder_ema = torch.nn.DataParallel(style_encoder_ema, device_ids=device_ids, output_device=output_device)
    
    # F0_model = torch.nn.DataParallel(F0_model, device_ids=device_ids, output_device=output_device)
    # ASR_model = torch.nn.DataParallel(ASR_model, device_ids=device_ids, output_device=output_device)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    nets_lmk = Munch(lmk_encoder=lmk_encoder,
                     mot_decoder=mot_decoder,
                     discriminator_mot=discriminator_mot)

    return nets, nets_ema, nets_lmk

def build_model_trans_pca(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    lmk_encoder = LmkEncoderPCA()
    mot_decoder = MotDecoderPCATR()
    discriminator_mot = DiscriminatorMotPCA(args.num_domains)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    nets_lmk = Munch(lmk_encoder=lmk_encoder,
                     mot_decoder=mot_decoder,
                     discriminator_mot=discriminator_mot)

    return nets, nets_ema, nets_lmk
