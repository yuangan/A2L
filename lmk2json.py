import cv2
import numpy as np
import mediapipe as mp
import glob
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


input_img = cv2.imread('./projects/fs_vid2vid/test_data/faceForensics/reference/images/000000.jpg')
wid = input_img.shape[1]
hig = input_img.shape[0]
print(wid,hig)

outjson_path = './projects/fs_vid2vid/test_data/faceForensics/reference/landmarks-dlib68/000000.json'

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    lmks = face_mesh.process(input_img).multi_face_landmarks[0].landmark

    jdata = []

    for lmk in lmks:
        tdata = [ lmk.x * wid , lmk.y * hig ]
        jdata.append(tdata)
    

    with open(outjson_path,'w') as f:
        json.dump(jdata,f)
