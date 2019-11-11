
import os
x='datasets/ECSSD/images'

# def get_train_filenames(x):
tem=[]
with os.scandir(x) as entries:
    for entry in entries:
        if os.path.splitext(entry)[1] in ['.jpg']:
           a = 'datasets/ECSSD/images/'+entry.name
           # print(a)
           tem.append(a)


data={'ECSSD':tem}
