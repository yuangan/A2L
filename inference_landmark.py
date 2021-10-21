# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import os
import imageio

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder, LmkEncoder, MotDecoder, Discriminator

# %matplotlib inline
emo_label = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
# speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=800)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
#     print(wave_tensor.shape)
    mel_tensor = to_mel(wave_tensor)
#     print(mel_tensor.shape)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)

    lmk_encoder = LmkEncoder()
    mot_decoder = MotDecoder()

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    nets_lmk = Munch(lmk_encoder=lmk_encoder,
                     mot_decoder=mot_decoder)

    return nets_ema, nets_lmk

def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to('cuda')
            extend_mel = torch.zeros((1, 80, 192)).float().to('cuda')
            mel_size = mel_tensor.size(2)
            extend_mel[0, :, :mel_size] = mel_tensor

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(extend_mel.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

#visualize mlmks
from functions import cv_draw_landmark
def visual_landmarks(initlm, fakemotion, writer):
    fl = []
    fl.append(initlm[:,:,:2])
    # calculate fake landmarks
    for i in range(fakemotion.shape[1]):
        fl.append(fl[i] + fakemotion[:,i,:,:])
    fl = torch.cat(fl, 0).cpu().numpy()

    heatmap = 255*np.ones((256, 256, 3), dtype=np.uint8)
    fl[:, :, 0:2] = fl[:, :, 0:2] - [fl[0, 0, 0],-0.4]
    fl[:,:,0:2] = fl[:,:,0:2]*200
    fl = np.transpose(fl, (0, 2, 1))
    for l in fl:
        img_draw = cv_draw_landmark(heatmap, l)
        writer.append_data(img_draw[:, :, ::-1])
    writer.close()
    return fl

# load F0 model

F0_model = JDCNet(num_class=1, seq_len=200)
params = torch.load("Utils/JDC/bst.t7", map_location='cpu')['net']
F0_model.load_state_dict(params)
# _ = F0_model.eval()
F0_model = F0_model.to('cuda')

# load vocoder
from parallel_wavegan.utils import load_model
vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

# load starganv2

# model_path = '/home/gy/gy/benchmark/A2L_StarG/Models/A2L/epoch_00750.pth'
# model_mot_path = '/home/gy/gy/benchmark/A2L_StarG2/Models/A2L_mot/epoch_00750.pth'
model_mot_path = '/home/gy/gy/benchmark/A2L_StarG2/Models/A2L_mot_adv/epoch_03250.pth'
wav_path = '/data3/MEAD/M003/wav/surprised/level_3/022.wav'

# model_mot_path = '/home/gy/gy/benchmark/A2L_StarG2/Models/A2L_mot_adv/epoch_01000.pth'

with open('Models/A2L_mot/config_al.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2, models_mot = build_model(model_params=starganv2_config["model_params"])
# params = torch.load(model_path, map_location='cpu')
# params = params['model_ema']

params_mot = torch.load(model_mot_path, map_location='cpu')

params = params_mot['model']
params_mot = params_mot['model_mot']

_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
# _ = [starganv2[key].eval() for key in starganv2]
starganv2.style_encoder = starganv2.style_encoder.to('cuda')
starganv2.mapping_network = starganv2.mapping_network.to('cuda')
starganv2.generator = starganv2.generator.to('cuda')

_ = [models_mot[key].load_state_dict(params_mot[key]) for key in models_mot]
# _ = [models_mot[key].eval() for key in models_mot]
models_mot.lmk_encoder = models_mot.lmk_encoder.to('cuda')
models_mot.mot_decoder = models_mot.mot_decoder.to('cuda')

# load input wave
# selected_speakers = [236]
# k = random.choice(selected_speakers)
# print(k)
# wav_path = 'Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav'
source_emo = wav_path.split('/')[5]
# wav_path = '/data3/MEAD/M003/wav/surprised/level_3/022.wav'
# wav_path = '/data3/MEAD/M003/wav/disgusted/level_3/015.wav'
bname = os.path.basename(wav_path)
dsp = wav_path.replace('wav/', 'video/front/')
# print(dsp)
example_landmark_path = dsp.replace(bname, 'initlmk_'+bname.replace('wav', 'npy'))
example_landmark = torch.from_numpy(np.load(example_landmark_path)).unsqueeze(0).to('cuda')

motion_path= dsp.replace(bname, 'motion_'+bname.replace('wav', 'npy'))
motion_gt = torch.from_numpy(np.load(motion_path)).unsqueeze(0).to('cuda')[:,:,:,:2]
print('motion_gt: ', motion_gt.shape)
print('example_landmark_shape: ', example_landmark.shape)

# wav_path = '/data3/MEAD/M003/wav/angry/level_3/001.wav'
audio, source_sr = librosa.load(wav_path, sr=24000)
# audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32
# print('audio shape', audio.shape) #89600

# with reference, using style encoder
# speaker_dicts = {}
# for s in selected_speakers:
#     k = s
#     speaker_dicts['p' + str(s)] = ('Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))

speaker_dicts = {}
# speaker_dicts[f'sad'] = ('/data3/MEAD/M003/wav/sad/level_3/010.wav', emo_label.index('sad'))
# speaker_dicts[f'happy'] = ('/data3/MEAD/M003/wav/happy/level_3/001.wav', emo_label.index('happy'))

speaker_dicts[f'happy'] = ('', emo_label.index('happy'))
speaker_dicts[f'sad'] = ('', emo_label.index('sad'))
speaker_dicts[f'disgusted'] = ('', emo_label.index('disgusted'))
speaker_dicts[f'contempt'] = ('', emo_label.index('contempt'))
speaker_dicts[f'surprised'] = ('', emo_label.index('surprised'))
speaker_dicts[f'angry'] = ('', emo_label.index('angry'))
speaker_dicts[f'fear'] = ('', emo_label.index('fear'))


# speaker_dicts[f'surprised'] = (wav_path, emo_label.index('surprised'))
# speaker_dicts[f'fear'] = ('', emo_label.index('fear'))

# print(emo_label.index('sad'))
reference_embeddings = compute_style(speaker_dicts)

# conversion 
import time
start = time.time()

source = preprocess(audio).to('cuda')
extend_source = torch.zeros((1, 80, 192)).float().to('cuda')
mel_size = source.size(2)
extend_source[0, :, :mel_size] = source
print('source: ', extend_source.shape, source.shape)

keys = []
converted_samples = {}
reconstructed_samples = {}
converted_mels = {}

for key, (ref, _) in reference_embeddings.items():
    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(extend_source.unsqueeze(1))
#         print('infer 134 f0: ', f0_feat.shape) #torch.Size([1, 256, 10, 299])
        # print('f0_feat: ', f0_feat)
        out, audio_feature = starganv2.generator(extend_source.unsqueeze(1), ref, masks=None, F0=f0_feat)
        # assert(0)
        lm_feature = models_mot.lmk_encoder(example_landmark)
        motion_fake = models_mot.mot_decoder(lm_feature, audio_feature)
        b, length, _ = motion_fake.shape
        motion_fake = motion_fake.reshape(b,length,68,2)
        length = min(motion_fake.shape[1], motion_gt.shape[1])
        print(length)
        loss = F.mse_loss(motion_fake[:,:length,:,:], motion_gt[:,:length,:,:2])
        
        print(motion_fake[:,:length,:,:], motion_gt[:,:length,:,:2],loss)

        print(motion_fake.shape, loss)
        writer = imageio.get_writer('./motion_fake.mp4', fps=30)
        fake_fl = visual_landmarks(example_landmark, motion_fake[:,:length,:,:], writer)
        writer = imageio.get_writer('./motion_gt.mp4', fps=30)
        visual_landmarks(example_landmark, motion_gt, writer)


        np.save(f'./{source_emo}_{bname[:-4]}_to_{key}_fake', fake_fl)

        assert(0)
        
        c = out.transpose(-1, -2).squeeze().to('cuda')
        y_out = vocoder.inference(c)
        y_out = y_out.view(-1).cpu()
        print('y_out shape:', y_out.shape)
        if key not in speaker_dicts or speaker_dicts[key][0] == "":
            recon = None
        else:
            wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
            mel = preprocess(wave)
            c = mel.transpose(-1, -2).squeeze().to('cuda')
            recon = vocoder.inference(c)
            recon = recon.view(-1).cpu().numpy()

    converted_samples[key] = y_out.numpy()
    reconstructed_samples[key] = recon

    converted_mels[key] = out
    
    keys.append(key)
end = time.time()
print('total processing time: %.3f sec' % (end - start) )

import IPython.display as ipd
for key, wave in converted_samples.items():
    with open(f'converted_001_{key}.wav', 'wb') as f:
        print('Converted: %s' % key)
        f.write(ipd.Audio(wave, rate=24000).data)
        print('Reference (vocoder): %s' % key)
        if reconstructed_samples[key] is not None:
            with open(f'recon_{key}.wav', 'wb') as f:
                f.write(ipd.Audio(reconstructed_samples[key], rate=24000).data)

print('Original (vocoder):')
wave, sr = librosa.load(wav_path, sr=24000)
mel = preprocess(wave)
c = mel.transpose(-1, -2).squeeze().to('cuda')
with torch.no_grad():
    recon = vocoder.inference(c)
    recon = recon.view(-1).cpu().numpy()
with open(f'source_recon.wav', 'wb') as f:
    f.write(ipd.Audio(recon, rate=24000).data)
print('Original:')
with open(f'source.wav', 'wb') as f:
    f.write(ipd.Audio(wav_path, rate=24000).data)