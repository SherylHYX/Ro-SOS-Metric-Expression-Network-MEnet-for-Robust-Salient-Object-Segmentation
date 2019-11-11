import os
x='datasets/SOD/BSDS300-images/BSDS300/images/test'
y='datasets/SOD/BSDS300-images/BSDS300/images/train'

tem=[]
with os.scandir(x) as entries:
    for entry in entries:
        if os.path.splitext(entry)[1] in ['.jpg']:
            a = 'datasets/SOD/BSDS300-images/BSDS300/images/test/'+entry.name
            # print(a)
            tem.append(a)

with os.scandir(y) as entries:
    for entry in entries:
        if os.path.splitext(entry)[1] in ['.jpg']:
            b = 'datasets/SOD/BSDS300-images/BSDS300/images/train/'+entry.name
            # print(b)
            tem.append(b)

data={'SOD':tem}
