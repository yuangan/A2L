#coding: utf-8

import os
import time
import random
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 800
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 800
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 96

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label, initlmk, motion, input_lmks = self._load_data(data)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label, ref_initlmk, ref_motion, _ = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _, ref2_initlmk, ref2_motion, _ = self._load_data(ref2_data)
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label, initlmk, motion, input_lmks

    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        initlmk, motion, input_lmks = self._load_motion(path)

        # left track == right track in 5 samples
        if len(wave_tensor.shape) > 1:
            wave_tensor = wave_tensor[:, 0]
        
        ##TODO: Release when training -by gy.
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:

            random_start = np.random.randint(0, mel_length - self.max_mel_length)

            # random_start = 0
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            # motion = motion = [:, random_start]
            motion = motion[random_start:random_start + self.max_mel_length, :, :]
            input_lmks = input_lmks[random_start:random_start + self.max_mel_length, :, :]
            if random_start >= 1:
                for i in range(random_start-1):
                    initlmk = initlmk + motion[i, :, :]
        else:
            motion = motion[:mel_length-1, :, :]
            input_lmks = input_lmks[:mel_length-1, :, :]
        # print(mel_tensor.shape, motion.shape, input_lmks.shape)
        return mel_tensor, label, initlmk, motion, input_lmks

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, label

    def _load_motion(self, data):
        bname = os.path.basename(data[0])
        dsp = data[0].replace('wav/', 'video/front/')
        initlmk_path = dsp.replace(bname, 'initlmk_'+bname.replace('wav', 'npy'))
        initlmk = np.load(initlmk_path)
        motion_path= dsp.replace(bname, 'motion_'+bname.replace('wav', 'npy'))
        motion = np.load(motion_path)
        
        input_lmks = [initlmk]
        for i in motion:
            input_lmks.append(input_lmks[-1]+i)
        input_lmks = np.stack(input_lmks, axis=0)

        return torch.from_numpy(initlmk), torch.from_numpy(motion), torch.from_numpy(input_lmks)

class MelDatasetPCA(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 96

        self.mean_mead = torch.from_numpy(np.load('./PCA/mean_mead.npy'))
        self.U = torch.from_numpy(np.load('./PCA/U_mead.npy'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label, initlmk, motion, output_lmks = self._load_data(data)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label, ref_initlmk, ref_motion, _ = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _, ref2_initlmk, ref2_motion, _ = self._load_data(ref2_data)
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label, initlmk, motion, output_lmks

    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        initlmk, motion, output_lmks = self._load_motion(path)
        # left track == right track in 5 samples
        if len(wave_tensor.shape) > 1:
            wave_tensor = wave_tensor[:, 0]
        
        ##TODO: Release when training -by gy.
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        # print(initlmk.shape, output_lmks.shape, mel_tensor.shape, path)

        ##TODO: This is a bug
        if motion.shape[0] < (mel_tensor.shape[1]-2):
            # print(path, ' has problem.')
            # print('motion shape: ', motion.shape, 'mel_tensor shape: ', mel_tensor.shape)
            # assert(0)
            mel_length = min(motion.shape[0], mel_tensor.shape[1])
        else:
            mel_length = mel_tensor.size(1)
        
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)

            # random_start = 0
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            # motion = motion = [:, random_start]

            if random_start >= 1:
                for i in range(random_start-1):
                    initlmk = initlmk + motion[i, :, :]
            motion = motion[random_start : random_start + self.max_mel_length, :, :]
            output_lmks = output_lmks[random_start : random_start + self.max_mel_length, :]
        else:
            mel_tensor = mel_tensor[:, :mel_length-1]
            motion = motion[:mel_length-1, :, :]
            output_lmks = output_lmks[:mel_length-1, :]
        # print(mel_tensor.shape, motion.shape, input_lmks.shape)
        return mel_tensor, label, initlmk, motion, output_lmks

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        # print('wave: ', wave_tensor.shape, wave_path)
        return wave_tensor, label

    def _load_motion(self, data):
        bname = os.path.basename(data[0])
        dsp = data[0].replace('wav/', 'video/front/')
        # print(dsp)
        initlmk_path = dsp.replace(bname, 'initlmk_'+bname[:-4]+'_multiland.npy')
        initlmk = np.load(initlmk_path)
        # print(initlmk.shape)
        motion_path= dsp.replace(bname, 'motion_'+bname[:-4]+'_multiland.npy')
        motion = np.load(motion_path)
        input_lmks = [initlmk]
        # print('mot: ', motion.shape)
        for i in motion:
            input_lmks.append(input_lmks[-1]+i)
        output_lmks = np.stack(input_lmks[1:], axis=0)[:,:,:2]
        b = output_lmks.shape[0]
        output_lmks = np.reshape(output_lmks, [b, 468*2])
        output_lmks = torch.from_numpy(output_lmks)

        # print(((input_lmks[1:] - input_lmks[0:-1]) == motion).any()==True)
        output_lmks = torch.mm(output_lmks - self.mean_mead.expand_as(output_lmks), self.U[:,:32])

        return torch.from_numpy(initlmk), torch.from_numpy(motion), output_lmks

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 96
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        inits = torch.zeros((batch_size, 68, 3))
        mots = torch.zeros((batch_size, self.max_mel_length, 68, 3))
        length_mots = torch.zeros((batch_size)).long()
        input_lmkss = torch.zeros((batch_size, self.max_mel_length, 68, 3))

        for bid, (mel, label, ref_mel, ref2_mel, ref_label, init, mot, input_lmks) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            # print('mels shape:', mels.shape, mel.shape)
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

            inits[bid] = init

            mot_size = min(mot.size(0), self.max_mel_length)
            length_mots[bid] = mot_size
            mots[bid,:mot_size,:,:] = mot[:mot_size,:,:]
            input_lmkss[bid,:mot_size,:,:] = input_lmks

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2, inits, mots, length_mots, input_lmkss

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

class CollaterPCA(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 96
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        inits = torch.zeros((batch_size, 468, 3))
        mots = torch.zeros((batch_size, self.max_mel_length, 468, 3))
        length_mots = torch.zeros((batch_size)).long()
        output_lmkss = torch.zeros((batch_size, self.max_mel_length, 32))

        for bid, (mel, label, ref_mel, ref2_mel, ref_label, init, mot, output_lmks) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            # print('mels shape:', mels.shape, mel.shape)
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

            inits[bid] = init

            mot_size = min(mot.size(0), self.max_mel_length)
            length_mots[bid] = mot_size
            mots[bid,:mot_size,:,:] = mot[:mot_size,:,:]
            output_lmkss[bid,:mot_size,:] = output_lmks

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2, inits, mots, length_mots, output_lmkss

def build_dataloader_pca(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDatasetPCA(path_list, validation=validation)
    collate_fn = CollaterPCA(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
