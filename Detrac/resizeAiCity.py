import cv2
import yolo.darknet as dn
import glob
from tqdm import tqdm
import random
dn.set_gpu(1)
Width = 10
# net = dn.load_net(str.encode("yolo/cfg/yolov3.cfg"),
#                       str.encode("yolo/checkpoint/yolov3.weights"), 0)
# meta = dn.load_meta(str.encode("yolo/cfg/coco.data"))

net = dn.load_net(str.encode("yolo/cfg/yolov3-detrac.cfg"),
                      str.encode("yolo/checkpoint/detrac/yolov3-detrac_final.weights"), 0)
meta = dn.load_meta(str.encode("./yolo/cfg/detrac.data"))
cnt = 0
for video_path in glob.glob('/home/zlz/DataSets/AI CITY CHALLENGE/aic19-track3-train-data/*.mp4'):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(7))
    for i in tqdm(random.sample(range(video_length - 1), 5000)):
        cap.set(1, i)
        ret, frame = cap.read()
        ret = dn.custom_detect(net, meta, frame)
        w_list = []
        if len(ret) != 0:
            f = open('SmallObjectTrainData/{}.txt'.format(cnt), 'w')
            for obj in ret:
                if obj[1] > 0.5:
                    bbox = obj[2]
                    [x, y, w, h] = bbox
                    print('0 {} {} {} {}'.format(x, y, w, h), file=f)
                    w_list.append(w)
                    # pt1 = (int(bbox[0] - 0.5 * bbox[2]), int(bbox[1] - 0.5 * bbox[3]))
                    # pt2 = (int(bbox[0] + 0.5 * bbox[2]), int(bbox[1] + 0.5 * bbox[3]))
                    # cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            f.close()
            w_mean = sum(w_list)/len(w_list)
            if w_mean > 300:
                continue
            alpha = w_mean / 10
            new_w = int(800//alpha)
            new_h = int(410//alpha)
            resize = cv2.resize(frame, (new_w, new_h))
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            cv2.imwrite('SmallObjectTrainData/{}.jpg'.format(cnt), resize)
            cnt += 1
