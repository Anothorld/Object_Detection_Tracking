import os
import glob
import random

jpg_list = glob.glob('/home/zlz/PycharmProjects/Detrac/SmallObjectTrainData/*.jpg')
test_list = random.sample(jpg_list, int(0.3 * len(jpg_list)))
data_file = open('SOD.data', 'w')
train = open('train_SOD.txt', 'w')
test = open('test_SOD.txt', 'w')

for img_path in jpg_list:
    if img_path in test_list:
        print(img_path, file=test)
    else:
        print(img_path, file=train)
    print('class = 4', file=data_file)
    print('train = {}'.format(glob.glob('./train_SOD.txt')), file=data_file)
    print('test = {}'.format(glob.glob('./test_SOD.txt')), file=data_file)
    print('backup = backup/SOD', file=data_file)