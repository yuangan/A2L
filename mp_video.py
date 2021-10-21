import cv2
import numpy as np
import mediapipe as mp
import glob
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# wait for process: "W016","W017","W018","W019","W023","W024","W025","W026","W028","W029","W033","W035","W036","W037","W038","W040"

# M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 W009 W011 W014 M042 W015
# M012 M013 M022 M026 M027 M031 M037 M041 W014
# W016 W018 W019 W023 W024 W025 W026 W028 W029 W033 W035 W036
# W040 W038 W037
in_path = glob.glob('/data3/MEAD/W036/video/front/*/level_*/0*.mp4')
#in_path = glob.glob('/data3/MEAD/M012/video/front/disgusted/level_2/027.mp4')
#print(in_path)
out_path = []
out_path_initlmk = []
out_path_motion = []
for pid,path in enumerate(in_path): 
    #print(pid,path)
    p,f = os.path.split(path)
    na,ext = os.path.splitext(f)
    #print(p+"/"+na+"_multiland.npy")
    out_path.append(p+"/"+na+"_multiland.npy")
    out_path_initlmk.append(p+"/"+na+"_initlmk_multiland.npy")
    out_path_motion.append(p+"/"+na+"_motion_multiland.npy")


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    for vid,vpath in enumerate(in_path):
        videoReader = cv2.VideoCapture(in_path[vid])
        fs = videoReader.get(cv2.CAP_PROP_FPS)
        sz = (int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #vw = cv2.VideoWriter('./output/video.mp4',cv2.VideoWriter_fourcc('M','P','E','G'), fs, sz)

        land_res = [] # 帧数 * 3 * landmark数量
        motion_res = []
        initlmk_res = []

        success, frame = videoReader.read()  
        idx = 0
        k = 0
        while success: 
            #print(success)
            #print(k)
            k += 1
            image = frame.copy()
            #cv2.imwrite("./imgs/"+str(k)+"_im.png",image)

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                success, frame = videoReader.read() #获取下一帧
                continue

            face_landmarks = results.multi_face_landmarks[0]
            land_loc = []
            xlis = []
            ylis = []   
            zlis = []
            for lm in face_landmarks.landmark:
                x = lm.x * sz[0]
                y = lm.y * sz[1]
                xlis.append(x)
                ylis.append(y)
                zlis.append(lm.z)
                #print(x,y,lm.z)
            land_loc.append(xlis)       
            land_loc.append(ylis)
            land_loc.append(zlis)
            land_res.append(land_loc)
            if idx == 0 : initlmk_res.append(land_loc)
            motion_res.append( list(   np.array(land_loc) - np.array(land_res[ len(land_res) - 1 ])  ) )
            idx += 1

            # for face_landmarks in results.multi_face_landmarks:
            #     mp_drawing.draw_landmarks(
            #         image=image,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_CONTOURS,
            #         landmark_drawing_spec=drawing_spec,
            #         connection_drawing_spec=drawing_spec)

            #cv2.imwrite('./output/video' + str(idx) + '.png', image)
            
            #vw.write(image)   # 写视频帧 
            success, frame = videoReader.read() #获取下一帧
        
        videoReader.release()
        #vw.release()

        res = np.array(land_res)
        np.save(out_path[vid],res)
        #np.save(out_path_initlmk[vid],initlmk_res)
        #np.save(out_path_motion[vid],motion_res)
        print("out:"+out_path[vid])


