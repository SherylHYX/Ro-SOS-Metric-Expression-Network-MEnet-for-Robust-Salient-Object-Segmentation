import os
x='datasets/DUT/DUT-OMRON-image'

tem=[]
with os.scandir(x) as entries:
    for entry in entries:
        if os.path.splitext(entry)[1] in ['.jpg']:
            a = 'datasets/DUT/DUT-OMRON-image/'+entry.name
            # print(a)
            tem.append(a)

data={'DUT':tem}
