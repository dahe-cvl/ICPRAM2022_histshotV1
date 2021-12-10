import numpy as np
import os
import shutil
import json
import cv2

root_db = "./movienet/trailer/"
dst_path = "./movienet/all_movienet_shottypes_v2/"
class_names = ["D0", "D1", "CU", "D3", "MS", "D5", "LS", "ELS", "D8", "D9"]

annotation_path = root_db + "annotations/v1_full_trailer.json"
data_path = root_db + "data/"

# read annotation file lists
f = open(annotation_path, 'r')
data = json.load(f)
film_list = data.keys()

entries_l = []
for vid in film_list:
    sid_list = data[vid].keys()
    for sid in sid_list:
        stc_label = data[vid][sid]['scale']['label']
        stc_label_idx = data[vid][sid]['scale']['value']
        cmc_label = data[vid][sid]['movement']['label']
        cmc_label_idx = data[vid][sid]['movement']['value']

        if(cmc_label == "Static"): # and ( stc_label == "ECS" or stc_label == "CS" or stc_label == "MS" or stc_label == "FS" or stc_label == "LS")):
            #print("-------------------")
            #print(vid)
            #print(sid)
            #print(stc_label)
            #print(cmc_label)
            entries_l.append([vid, sid, stc_label, cmc_label])
f.close()

entries_np = np.array(entries_l)
print(entries_np[:2])

# get center frame of shot
for i in range(0, len(entries_np)):
    
    vid_name = entries_np[i][0]
    sid_num = entries_np[i][1]
    class_name = entries_np[i][2]

    vid_path = data_path + "/" + vid_name
    sid_path = data_path + "/" + vid_name + "/shot_" + sid_num + ".mp4"

    cap = cv2.VideoCapture(sid_path)

    frames_l = []
    while(cap.isOpened()):
        ret, frame_np = cap.read()
        if ret == True:
            #cv2.imshow('Frame', frame_np)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    break
            frames_l.append(frame_np)
        else:
            break
    cap.release()
    frames_np = np.array(frames_l)

    shot_len = len(frames_np)
    center_frame_pos = int(shot_len / 2)
    center_frame_np = frames_np[center_frame_pos]
    print(center_frame_np.shape)

    final_dst_path = dst_path + "/" + class_name + "/" + vid_name + "_" + class_name + "_" + sid_num + ".png" 
    print(final_dst_path)
    cv2.imwrite(final_dst_path, center_frame_np)
