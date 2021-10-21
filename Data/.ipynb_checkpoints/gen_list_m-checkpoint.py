import os
import glob
wavs = glob.glob('/data3/MEAD/*/wav/*/level_*/*.wav')
muls = glob.glob('/data3/MEAD/*/video/front/*/level_*/initlmk*multiland.npy')
emo_label = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
val_list = ['001.wav', '002.wav', '003.wav']
with open('train_emo_list_new.txt', 'w') as f:
    for wav in wavs:
        flag_con = False
        for v in val_list:
            if v in wav:
                flag_con = True
                break
            if 'neutral' in wav:
                flag_con = False
                break
            if 'level_1' in wav or 'level_2' in wav:
                flag_con = True
                break
        if flag_con:
            continue
        emo_cls = wav.split('/')[-3]
        wavsp = (wav.split('/'))
#         print(emo_label.index(emo_cls))
        mul = '/data3/MEAD/' + wavsp[3] + '/video/front/' + wavsp[5] + '/' + wavsp[6] + '/initlmk_' + wavsp[7][:-4] + '_multiland.npy'
        if mul in muls:
            print('exist!')
            f.write(f'{wav}|{emo_label.index(emo_cls)}\n')

with open('val_emo_list_new.txt', 'w') as f:
    for wav in wavs:
        for v in val_list:
            if v in wav:
                wavsp = (wav.split('/'))
                mul = '/data3/MEAD/' + wavsp[3] + '/video/front/' + wavsp[5] + '/' + wavsp[6] + '/initlmk_' + wavsp[7][:-4] + '_multiland.npy'
                if mul in muls:
                    print('exist!')
                    emo_cls = wav.split('/')[-3]
                    print(wav)
            #         print(emo_label.index(emo_cls))
                    f.write(f'{wav}|{emo_label.index(emo_cls)}\n')