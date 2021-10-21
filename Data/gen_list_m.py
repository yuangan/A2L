import os
import glob
wavs = glob.glob('/data3/shared/MEAD/*/wav/*/level_3/*.wav')
wavs.extend(glob.glob('/data3/shared/MEAD/*/wav/neutral/level_1/*.wav'))

muls = glob.glob('/data3/shared/MEAD/*/video/front/*/level_*/initlmk*multiland.npy')
print(len(wavs))
print(len(muls))
emo_label = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
val_list = ['001.wav', '002.wav', '003.wav']
with open('train_emo_list_all.txt', 'w') as f:
    for wav in wavs:
        flag_con = False
        for v in val_list:
            if v in wav:
                flag_con = True
                break

        if flag_con:
            continue
        emo_cls = wav.split('/')[-3]
        wavsp = wav.split('/')
#         print(emo_label.index(emo_cls))
        mul = '/data3/shared/MEAD/' + wavsp[4] + '/video/front/' + wavsp[6] + '/' + wavsp[7] + '/initlmk_' + wavsp[8][:-4] + '_multiland.npy'

        if mul in muls:
            f.write(f'{wav}|{emo_label.index(emo_cls)}\n')
        else:
            print(mul)

with open('val_emo_list_all.txt', 'w') as f:
    for wav in wavs:
        for v in val_list:
            if v in wav:
                wavsp = (wav.split('/'))
                mul = '/data3/shared/MEAD/' + wavsp[4] + '/video/front/' + wavsp[6] + '/' + wavsp[7] + '/initlmk_' + wavsp[8][:-4] + '_multiland.npy'
                if mul in muls:
                    emo_cls = wav.split('/')[-3]
                    # print(wav)
                    f.write(f'{wav}|{emo_label.index(emo_cls)}\n')
                else:
                    if 'level_1' not in mul and 'level_2' not in mul:
                        print(wav)