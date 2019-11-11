import os
x='datasets/MSRA1000/Images'

tem=[]
with os.scandir(x) as entries:
    for entry in entries:
        if os.path.splitext(entry)[1] in ['.jpg']:
            a = 'datasets/MSRA1000/Images/'+entry.name
            # print(a)
            tem.append(a)

data={'MSRA1000':tem}
