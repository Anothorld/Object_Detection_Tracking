import os
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import random

WIDTH = 960
HEIGHT = 540


def one_xml2txt(xml_file, img_root=None):
    FrameNumCount = 1
    video_name = xml_file.split('/')[-1].split('.')[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ignore = root.find('ignored_region')
    ignore_boxs = ignore.findall('box')
    ignore_area = []
    for k, ign_box in enumerate(ignore_boxs):
        x = int(float(ign_box.attrib['left']))
        y = int(float(ign_box.attrib['top']))
        w = int(float(ign_box.attrib['width']))
        h = int(float(ign_box.attrib['height']))
        ignore_area.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    childs_obj = root.findall('frame')
    for child_frame in tqdm(childs_obj):
        box_list = []
        frameID = int(child_frame.attrib['num'])
        density = int(child_frame.attrib['density'])
        labels = np.array([b'Vehicle' for i in range(density)])
        # img = cv2.imread(os.path.join(img_root, 'img{:05}.jpg'.format(frameID)))
        # cv2.fillPoly(img, ignore_area, (127, 127, 127))
        # # cv2.imshow('img', img)
        # # cv2.waitKey(0)
        # cv2.imwrite('/home/zlz/PycharmProjects/Detrac/YOLOTrainData/img{:05}.jpg'.format(FrameNumCount), img)
        # print('frameID: ', frameID)
        target_list = child_frame.find('target_list')
        child1s_target = target_list.findall('target')
        for child1_target in child1s_target:
            targetID = int(child1_target.attrib['id'])
            box = child1_target.find('box').attrib
            box_list.append([float(box['left']), float(box['top']), float(box['left']) + float(box['width']),
                          float(box['top']) + float(box['height'])])
        boxes = np.array(box_list, dtype=np.float32)
        np.savez('{}_F_{:08}.npz'.format(video_name, FrameNumCount), labels=labels, boxes=boxes)
        FrameNumCount += 1

def xml2txt():
    label_list = sorted(glob.glob('/home/zlz/DataSets/DETRAC/DETRAC-Train-Annotations-XML/*.xml'),
                        key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
    FrameNumCount = 1
    for label_path in label_list:
        box_list = []
        print(label_path)
        name = label_path.split('/')[-1].split('.')[0]
        img_root = os.path.join('/home/zlz/DataSets/DETRAC/DETRAC-train-data/Insight-MVT_Annotation_Train', name)
        tree = ET.parse(label_path)
        root = tree.getroot()
        ignore = root.find('ignored_region')
        ignore_boxs = ignore.findall('box')
        ignore_area = []
        for k, ign_box in enumerate(ignore_boxs):
            x = int(float(ign_box.attrib['left']))
            y = int(float(ign_box.attrib['top']))
            w = int(float(ign_box.attrib['width']))
            h = int(float(ign_box.attrib['height']))
            ignore_area.append(np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]))
        childs_obj = root.findall('frame')
        for child_frame in tqdm(childs_obj):
            boxes = []
            frameID = int(child_frame.attrib['num'])
            density = int(child_frame.attrib['density'])
            labels = np.array([b'Vehicle' for i in range(density)])
            img = cv2.imread(os.path.join(img_root, 'img{:05}.jpg'.format(frameID)))
            cv2.fillPoly(img, ignore_area, (127, 127, 127))
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            cv2.imwrite('/home/zlz/PycharmProjects/Detrac/YOLOTrainData/img{:05}.jpg'.format(FrameNumCount), img)
            f = open('/home/zlz/PycharmProjects/Detrac/YOLOTrainData/img{:05}.txt'.format(FrameNumCount), 'w')
            # print('frameID: ', frameID)
            target_list = child_frame.find('target_list')
            child1s_target = target_list.findall('target')
            for child1_target in child1s_target:
                targetID = int(child1_target.attrib['id'])
                box = child1_target.find('box').attrib
                vehicle_class = child1_target.find('attribute').attrib['vehicle_type']
                if vehicle_class == 'car':
                    classID = 0
                elif vehicle_class == 'bus':
                    classID = 1
                elif vehicle_class == 'van':
                    classID = 2
                else:
                    classID = 3
                print('{} {:.5} {:.5} {:.5} {:.5}'.format(classID, (float(box['left']) + 0.5*float(box['width']))/ WIDTH,
                                                          (float(box['top']) + 0.5*float(box['height'])) / HEIGHT,
                                                                   float(box['width']) / WIDTH,
                                                                   float(box['height']) / HEIGHT), file=f)
            f.close()
            FrameNumCount += 1


# generate detrac.data and detrac.names and tain or test.text
def generate_cfgs():
    train = open('train.txt', 'w')
    test = open('test.txt', 'w')
    # test_root = '/home/zlz/DataSets/DETRAC/DETRAC-test-data/Insight-MVT_Annotation_Test'
    trainlist = sorted(glob.glob('/home/zlz/PycharmProjects/Detrac/YOLOTrainData/*.jpg'),
                      key=lambda x: int(x.split('/')[-1].split('.')[0].split('img')[-1]))
    # testdir = os.listdir(test_root)
    # for img_dir in testdir:
    #     for img_path in glob.glob(os.path.join(test_root, img_dir, '*.jpg')):
    #         print(img_path, file=test)
    length = len(trainlist)
    test_index = random.sample(trainlist, int(0.3 * length))
    for i, txt in enumerate(trainlist):
        if txt not in test_index:
            print(txt, file=train)
        else:
            print(txt, file=test)
    train.close()
    test.close()
    data = open('detrac.data', 'w')
    print('class = 4', file=data)
    print('train = /home/zlz/PycharmProjects/Detrac/train.txt', file=data)
    print('train = /home/zlz/PycharmProjects/Detrac/test.txt', file=data)
    print('names = /home/zlz/PycharmProjects/Detrac/detrac.names', file=data)
    print('backup = backup/detrac', file=data)

if __name__ == "__main__":
    one_xml2txt('/home/arnold/PycharmProjects/Object_Detection_Tracking/MVI_20011.xml')
    # xml2txt()
    # generate_cfgs()