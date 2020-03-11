import cv2
import random
import numpy as np
import yolo.darknet as dn

W = 960
H = 540

dn.set_gpu(0)
net = dn.load_net(str.encode("yolo/cfg/yolov3-detrac.cfg"),
                      str.encode("yolo/checkpoint/detrac/yolov3-detrac_final.weights"), 0)
meta = dn.load_meta(str.encode("./yolo/cfg/detrac.data"))
dn.set_gpu(1)
net1 = dn.load_net(str.encode("yolo/cfg/yolov3.cfg"),
                      str.encode("yolo/checkpoint/yolov3.weights"), 0)
meta1 = dn.load_meta(str.encode("yolo/cfg/coco.data"))

path_list = open('/home/zlz/PycharmProjects/Detrac/train.txt', 'r').readlines()

for i, img_path in enumerate(random.sample(path_list, 200)):
    img = cv2.imread(img_path[0:-1])
    img_train = img.copy()
    img_yolo = img.copy()
    img_label = img.copy()
    ret = dn.custom_detect(net, meta, img)
    ret1 = dn.custom_detect(net1, meta1, img)
    label = open(img_path[0:-4]+'txt', 'r')
    for line in label.readlines():
        bb_lsit = line.split(' ')[1:]
        bbox = [float(i) for i in bb_lsit]
        bbox[0] = bbox[0]*W
        bbox[2] = bbox[2] * W
        bbox[1] = bbox[1] * H
        bbox[3] = bbox[3] * H
        pt1 = (int(bbox[0] - 0.5 * bbox[2]), int(bbox[1] - 0.5 * bbox[3]))
        pt2 = (int(bbox[0] + 0.5 * bbox[2]), int(bbox[1] + 0.5 * bbox[3]))
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
        cv2.rectangle(img_label, pt1, pt2, (0, 0, 255), 2)
        cv2.putText(img, "R:Label", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "B:Trained", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, "G:Untrained", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for obj in ret:
        bbox = obj[2]
        pt1 = (int(bbox[0] - 0.5*bbox[2]), int(bbox[1] - 0.5*bbox[3]))
        pt2 = (int(bbox[0] + 0.5*bbox[2]), int(bbox[1] + 0.5*bbox[3]))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
        cv2.rectangle(img_train, pt1, pt2, (255, 0, 0), 2)
    for obj in ret1:
        if obj[0] == b'car' or obj[0] == b'bus' or obj[0] == b'truck':
            if obj[1] > 0.75:
                bbox = obj[2]
                pt1 = (int(bbox[0] - 0.5*bbox[2]), int(bbox[1] - 0.5*bbox[3]))
                pt2 = (int(bbox[0] + 0.5*bbox[2]), int(bbox[1] + 0.5*bbox[3]))
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                cv2.rectangle(img_yolo, pt1, pt2, (0, 255, 0), 2)
        else:
            continue
    top = np.concatenate([img, img_train], axis=1)
    bottom = np.concatenate([img_label, img_yolo], axis=1)
    result = np.concatenate([top, bottom])
    result = cv2.resize(result, (1440, 810))
    cv2.imwrite('Result/{}.jpg'.format(i), result)
    # cv2.waitKey(0)