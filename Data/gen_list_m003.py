import os
import glob
wavs = glob.glob('/data3/MEAD/*/wav/*/level_*/*.wav')
emo_label = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
with open('train_emo_list.txt', 'w') as f:
    for wav in wavs:
#         if '2' in wav:
#             continue
        emo_cls = wav.split('/')[-3]
        print(wav)
#         print(emo_label.index(emo_cls))
        f.write(f'{wav}|{emo_label.index(emo_cls)}\n')