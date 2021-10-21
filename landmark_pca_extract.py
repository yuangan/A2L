from glob import glob
import torch
import numpy as np
import sys
sys.path.append('../3DDFA_V2-master/')
from utils.functions import cv_draw_landmark_468
import imageio
files = glob('/data3/MEAD/*/video/front/*/*/init*multiland.npy')

#visualize mlmks
def visual_landmarks(fl):
    writer = imageio.get_writer('./visuallmk.mp4', fps=30)
    heatmap = 255*np.ones((256, 256, 3), dtype=np.uint8)
    print(fl[0, 0, 0],fl[0, 0, 1]*500)
    fl[:, :, 0:2] = fl[:, :, 0:2] - [-0.4 ,-0.4]
    fl[:,:,0:2] = fl[:,:,0:2]*200
    fl = np.transpose(fl, (0, 2, 1))
    for l in fl:
        img_draw = cv_draw_landmark_468(heatmap, l)
        writer.append_data(img_draw[:, :, ::-1])
    writer.close()
    

def get_lmk(initlmk, motionlmk):
    # print(initlmk.shape, motionlmk.shape)
    lmks = [initlmk]
    for i in motionlmk:
        lmks.append(lmks[-1]+i)
    lmks = np.stack(lmks, axis=0)
    # print(lmks.shape)
    # visual_landmarks(lmks)
    return lmks

def extract_pca():
    landmarks = []
    for index in range(len(files)):
        # print(index, files[index])
        if index == 1500:
            break
        init_lmark = np.load(files[index])
        motion_lmark = np.load(files[index].replace('initlmk','motion'))
        lmks = get_lmk(init_lmark, motion_lmark)
        landmarks.append(torch.from_numpy(lmks))
    

    landmarks = torch.cat(landmarks, dim= 0)[:,:,0:2]
    landmarks = landmarks.reshape(landmarks.size(0), 468*2)
    mean = torch.mean(landmarks,0)
    print(mean)
    np.save('./PCA/mean_mead.npy', mean.numpy())
    landmarks = landmarks - mean.expand_as(landmarks)

    U,S,V  = torch.svd(torch.t(landmarks))
    print (S)
    print (U[:,:10])
    print(U.shape)
    # C = torch.mm(landmarks, U[:,:k])
    np.save('./PCA/U_mead.npy', U.numpy())
    np.save('./PCA/S_mead.npy', S.numpy())

extract_pca()