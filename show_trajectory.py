import cv2
import os
import numpy as np

INTERESTED_CLS = "Vehicle"

dir_path = '/home/arnold/PycharmProjects/Object_Detection_Tracking/test_track_out/cam_2.mp4'
# video_list = os.listdir('/home/arnold/PycharmProjects/Object_Detection_Tracking/AICITY_test_track_out')
# for dir_path in video_list:
print("Working on {}\n".format(dir_path))
cap = cv2.VideoCapture(os.path.join(os.getcwd(), "track1", dir_path.split('/')[-1]))
track_txt = open(os.path.join(dir_path, INTERESTED_CLS, dir_path.split('/')[-1].split('.')[0] + '.txt'), 'r')
object_buf = {}
color_buf = {}
delet_buf = []
ret, frame = cap.read()
last_frame_ind = 0
for line in track_txt.readlines():
    line_list = line.split('\n')[0].split(',')
    frame_ind = int(line_list[0])
    if line_list[1] not in color_buf:
        color_buf[line_list[1]] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
    if frame_ind != last_frame_ind:
        print("current object frame index", frame_ind)
        for object_ID in object_buf:
            if object_buf[object_ID][0] == last_frame_ind:
                current_frame = cv2.rectangle(frame, tuple([int(i) for i in object_buf[object_ID][1:3]]),
                              tuple([int(i + j) for i ,j in zip(object_buf[object_ID][1:3],
                                        object_buf[object_ID][3:])]), color_buf[object_ID])
                current_frame = cv2.putText(current_frame, "#{}".format(object_ID),
                                            (tuple([int(i)-10 for i in object_buf[object_ID][1:3]])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_buf[object_ID], 1)
            if object_buf[object_ID][0] != last_frame_ind and last_frame_ind >= 1:
                delet_buf.append(object_ID)
        for key in delet_buf:
            object_buf.pop(key)
            color_buf.pop(key)
        delet_buf = []
        cv2.imshow('current', current_frame)
        cv2.waitKey(0)
        if cap.get(1) != frame_ind:
            cap.set(1, frame_ind)
        ret, frame = cap.read()
        print("current frame index:", cap.get(1))
    object_buf[line_list[1]] = [frame_ind] + [float(i) for i in line_list[2:6]]
    last_frame_ind = frame_ind