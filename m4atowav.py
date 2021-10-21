
# from os import path
# from pydub import AudioSegment
# import glob
# # files                                                                   
# m4as = glob.glob('/data3/MEAD/M003/audio/angry/level_3/001.m4a')
# for src in m4as:
#     dst = src.replace('m4a', 'wav')

#     # convert wav to mp3                                                            
#     sound = AudioSegment.from_m4a(src)
#     sound.export(dst, format="wav")

import os
import glob
m4as = glob.glob('/data3/MEAD/*/audio/*/*/*.m4a')
for m4a in m4as:
    wav_path = m4a.replace('audio', 'wav').replace('m4a', 'wav')
    
    print(wav_path)
    if not os.path.exists(os.path.dirname(wav_path)):
        os.makedirs(os.path.dirname(wav_path))
#     assert(0)
    if not os.path.exists(wav_path):
        os.system(f'ffmpeg -i {m4a} -acodec pcm_s16le -ac 2 -ar 24000 {wav_path}')
    else:
        print('exist')