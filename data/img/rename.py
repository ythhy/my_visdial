import os

num = 0
directory = '/home/yth/visDial/data/img/VisualDialog_train2018'
for file in os.listdir(directory):
    path = os.path.join(directory, file)
    new_name = 'VisualDialog_train2018_'+file[-16:]
    target = os.path.join(directory, new_name)
    os.rename(path, target)
    num = num + 1
print(num)