#coding:utf-8

import os
import torch

from torch import nn
from munch import Munch
from transforms import build_transforms

import torch.nn.functional as F
import numpy as np

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False, use_con_reg=False):
    args = Munch(args)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out, x_real)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    if use_con_reg:
        t = build_transforms()
        out_aug = nets.discriminator(t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # with fake audios
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            
        F0 = nets.f0_model.module.get_feature_GAN(x_real)
        x_fake = nets.generator(x_real, s_trg, masks=None, F0=F0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.module.classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        if use_con_reg:
            out_de_aug = nets.discriminator.module.classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
        
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg 

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, use_adv_cls=False):
    args = Munch(args)
    
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
        
    # compute style vectors
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
    
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.module.get_feature(x_real)
    
    # adversarial loss
    x_fake = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    
    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
    ASR_fake = nets.asr_model.module.get_feature(x_fake)
    
    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - args.norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)
    
    # style F0 loss (style initialization)
    if x_refs is not None and args.lambda_f0_sty > 0 and not use_adv_cls:
        F0_sty, _, _ = nets.f0_model(x_ref)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = torch.zeros(1).mean()
    
    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)
    
    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    
    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())
         
    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None, F0=GAN_F0_fake)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    # F0 loss in cycle-consistency loss
    if args.lambda_f0 > 0:
        _, _, cyc_F0_rec = nets.f0_model(x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
    if args.lambda_asr > 0:
        ASR_recon = nets.asr_model.module.get_feature(x_rec)
        loss_cyc += F.smooth_l1_loss(ASR_recon, ASR_real)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.module.classifier(x_fake)

        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_trg[y_org != y_trg])
    else:
        loss_adv_cls = torch.zeros(1).mean()
    
    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc\
           + args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_adv_cls * loss_adv_cls

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item())
    

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

### TODO: add input_lmks to replace el when encoding lmk feature

def compute_lmk_g_loss(nets, net_mot, example_landmark, motion_gt, length_mots, input_lmks, x_real, y_org, y_trg, z_trgs=None, x_refs=None, use_r1_reg=True, use_adv_cls=False, use_adv=False):
    # args = Munch(args)
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # with real audios for mse loss(AE loss)
    # if use_audio_net:
    with torch.no_grad():
        if z_trgs is not None:
            s_trg = nets.mapping_network(z_trg, y_org)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_real, y_org)
        F0 = nets.f0_model.get_feature_GAN(x_real)
        _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
    lm_feature = net_mot.lmk_encoder(input_lmks)
    motion_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
    b, length, _ = motion_fake.shape
    motion_fake = motion_fake.reshape(b,length,68,2)
    mseloss = torch.FloatTensor([0]).to(x_real.device)
    for bid, length in enumerate(length_mots):
        mseloss += F.mse_loss(motion_fake[bid,:length,:,:], motion_gt[bid,:length,:,:2])
    mseloss = mseloss / len(length_mots)


    loss_adv = torch.FloatTensor([0]).to(x_real.device)
    if use_adv:

        # with fake audios
        with torch.no_grad():
            if z_trgs is not None:
                s_trg = nets.mapping_network(z_trg, y_trg)
            else:
                s_trg = nets.style_encoder(x_ref, y_trg)
            F0 = nets.f0_model.get_feature_GAN(x_real)
            _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
        lm_feature = net_mot.lmk_encoder(input_lmks)
        motion_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
        b, length, _ = motion_fake.shape
        motion_fake = motion_fake.reshape(b, length, 68, 2)
        out = net_mot.discriminator_mot(example_landmark, motion_fake, length_mots)
        loss_adv = adv_loss(out, 1)

    loss = 10*mseloss/length_mots.shape[0] + loss_adv
    return loss, Munch(motion=mseloss.item(),
                       adv=loss_adv.item())

def compute_lmk_d_loss(nets, net_mot, example_landmark, motion_gt, length_mots, input_lmks, x_real, y_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False):
    loss_real = torch.FloatTensor([0]).to(x_real.device)
    loss_fake = torch.FloatTensor([0]).to(x_real.device)
    loss_real_adv_cls = torch.FloatTensor([0]).to(x_real.device)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = net_mot.discriminator_mot(example_landmark, motion_gt, length_mots)
    loss_real = adv_loss(out, 1)
    
    # # with fake audios
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            
        F0 = nets.f0_model.get_feature_GAN(x_real)
        _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
    lm_feature = net_mot.lmk_encoder(input_lmks)
    motion_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
    b, l, _ = motion_fake.shape
    motion_fake = motion_fake.reshape(b,l,68,2)
    out = net_mot.discriminator_mot(example_landmark, motion_fake, length_mots)
    loss_fake = adv_loss(out, 0)
    # if use_con_reg:
    #     out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
    #     loss_con_reg += F.smooth_l1_loss(out, out_aug)
    use_adv_cls= True
    # # adversarial classifier loss
    if use_adv_cls:
        out_de = net_mot.discriminator_mot.classifier(example_landmark, motion_fake, length_mots)
        # print(out_de.shape, y_org.shape, y_org != y_trg)
        # print(out_de)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        # if use_con_reg:
        #     out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
        #     loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
    # assert(0)

    # # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    # if use_r1_reg:
    #     loss_reg = r1_reg(out, x_real)
    # else:
    #     loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    # loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    # if use_con_reg:
    #     t = build_transforms()
    #     out_aug = nets.discriminator(t(x_real).detach(), y_org)
    #     loss_con_reg += F.smooth_l1_loss(out, out_aug)
    

        
    loss = loss_real + loss_fake + loss_real_adv_cls
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       real_adv_cls=loss_real_adv_cls.item())

def compute_pcalmk_g_loss(nets, net_mot, example_landmark, motion_gt, length_mots, pca_lmks, x_real, y_org, y_trg, z_trgs=None, x_refs=None, use_r1_reg=True, use_adv_cls=False, use_adv=False, eval=False):
    # args = Munch(args)
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # with real audios for mse loss(AE loss)
    # if use_audio_net:
    # with torch.no_grad():
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_org)
    else:  # x_ref is not None
        s_trg = nets.style_encoder(x_real, y_org)
    F0 = nets.f0_model.get_feature_GAN(x_real)
    _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
    # print(x_real.shape, audio_feature.shape)
    
    lm_feature = net_mot.lmk_encoder(example_landmark)
    motion_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
    b, length, _ = motion_fake.shape
    motion_fake = motion_fake.reshape(b, length, 32)
    mseloss = torch.FloatTensor([0]).to(x_real.device)
    for bid, l in enumerate(length_mots):
        mseloss = mseloss + F.mse_loss(motion_fake[bid,:l,:], pca_lmks[bid,:l,:])
#         mseloss = mseloss + F.l1_loss(motion_fake[bid,:l,:], pca_lmks[bid,:l,:])
        # print(bid, length)
    
    mseloss = mseloss / len(length_mots)
    # print(mseloss)
    # assert(0)


    loss_adv = torch.FloatTensor([0]).to(x_real.device)
    if use_adv:

        # with fake audios
        #with torch.no_grad():
        if z_trgs is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)
        F0 = nets.f0_model.get_feature_GAN(x_real)
        _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
        lm_feature = net_mot.lmk_encoder(example_landmark)
        motion_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
        out = net_mot.discriminator_mot(example_landmark, y_trg, motion_fake, length_mots)
        loss_adv = adv_loss(out, 1)

    loss = 10*mseloss + loss_adv
    if not eval:
        return loss, Munch(recon=mseloss.item(),
                            adv=loss_adv.item())
    else:
        return loss, Munch(recon=mseloss.item(),
                            adv=loss_adv.item()), motion_fake.reshape(b, length, 32)

def compute_pcalmk_d_loss(nets, net_mot, example_landmark, motion_gt, length_mots, pca_lmks, x_real, y_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False):
    loss_real = torch.FloatTensor([0]).to(x_real.device)
    loss_fake = torch.FloatTensor([0]).to(x_real.device)
    loss_real_adv_cls = torch.FloatTensor([0]).to(x_real.device)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = net_mot.discriminator_mot(example_landmark, y_org, pca_lmks, length_mots)
    loss_real = adv_loss(out, 1)
    
    # # with fake audios
    #with torch.no_grad():
    if z_trg is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:  # x_ref is not None
        s_trg = nets.style_encoder(x_ref, y_trg)
            
    F0 = nets.f0_model.get_feature_GAN(x_real)
    _, audio_feature = nets.generator(x_real, s_trg, masks=None, F0=F0)
    
    lm_feature = net_mot.lmk_encoder(example_landmark)
    pca_fake = net_mot.mot_decoder(lm_feature, audio_feature, length_mots)
    out = net_mot.discriminator_mot(example_landmark, y_trg, pca_fake, length_mots)
    loss_fake = adv_loss(out, 0)
    # if use_con_reg:
    #     out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
    #     loss_con_reg += F.smooth_l1_loss(out, out_aug)
    # use_adv_cls= True
    # # adversarial classifier loss
    if use_adv_cls:
        out_de = net_mot.discriminator_mot.classifier(example_landmark, pca_fake, length_mots)
        # print(out_de.shape, y_org.shape, y_org != y_trg, y_trg.shape)
        # print(out_de)
        # print(out_de[y_org != y_trg], y_org[y_org != y_trg])

        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        # if use_con_reg:
        #     out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
        #     loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
    # assert(0)

    # # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    # if use_r1_reg:
    #     loss_reg = r1_reg(out, x_real)
    # else:
    #     loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    # loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    # if use_con_reg:
    #     t = build_transforms()
    #     out_aug = nets.discriminator(t(x_real).detach(), y_org)
    #     loss_con_reg += F.smooth_l1_loss(out, out_aug)
    

        
    loss = loss_real + loss_fake + loss_real_adv_cls
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       real_adv_cls=loss_real_adv_cls.item())
