import cv2
import os
import yaml
from ultralytics import YOLO
from utils import *

def baseball_interpolation(list):

    for i in range(len(list)):
        if list[i] != -1:
            l = i
            break

    for i in range(l + 1, len(list)):
        
        if list[i] != -1 and list[i - 1] == -1:
            sublist = interpolate_sublist(list[l:i + 1])
            list[l:i + 1] = sublist
        
        if list[i] != -1:
            l = i

    return list

def interpolate_sublist(sublist):
    
    if len(sublist) < 2:
        return sublist

    for i in range(1, len(sublist) - 1):
        left = i
        right = len(sublist) - i - 1
        sublist[i] = sublist[0] * right / (left + right) + sublist[-1] * left / (left + right)

    return sublist

if __name__ == "__main__":

    config = yaml.safe_load(open("config.yaml"))

    # Init
    model = YOLO(config["model"])
    class_name_dict = {0: "baseball"}
    color_dict = {0: (225, 25, 102), 1: (102, 210, 13), 2: (15, 185, 255)}
    detection_threshold = config["yolo_threshold"]

    video_folder = config["video_folder"]
    videos = os.listdir(video_folder)
    # print(videos)

    # Projection matrices
    P1 = get_projection_matrix(1, config["camera_parameters"] + "/")
    P2 = get_projection_matrix(2, config["camera_parameters"] + "/")
    P3 = get_projection_matrix(3, config["camera_parameters"] + "/")
    P4 = get_projection_matrix(4, config["camera_parameters"] + "/")

    # Read videos
    cap1 = cv2.VideoCapture(os.path.join(video_folder, videos[0]))
    cap2 = cv2.VideoCapture(os.path.join(video_folder, videos[1]))
    cap3 = cv2.VideoCapture(os.path.join(video_folder, videos[2]))
    cap4 = cv2.VideoCapture(os.path.join(video_folder, videos[3]))
    caps = [cap1, cap2, cap3, cap4]

    # Predict bounding boxes
    frame_3ds = []
    while True:
        
        is_detect_1 = False
        is_detect_2 = False
        is_detect_3 = False
        is_detect_4 = False

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        ''' ------------ '''
        ''' FIRST CAMERA '''
        ''' ------------ '''
        results = model(frame1)[0]

        if len(results.boxes.data.tolist()) > 0:
            index = 0
            max_score = -1
            max_index = -1
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > max_score:
                    max_score = score
                    max_index = index                    
                
            x1, y1, x2, y2, score, class_id = results.boxes.data.tolist()[max_index]
                    
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            if score > detection_threshold:
                is_detect_1 = True
                cam1_kpts2d = ((int((x1+x2)/2), int((y1+y2)/2)))

        ''' ------------- '''
        ''' SECOND CAMERA '''
        ''' ------------- '''
        results = model(frame2)[0]

        if len(results.boxes.data.tolist()) > 0:
            index = 0
            max_score = -1
            max_index = -1
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > max_score:
                    max_score = score
                    max_index = index                    
                
            x1, y1, x2, y2, score, class_id = results.boxes.data.tolist()[max_index]
                    
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            if score > detection_threshold:
                is_detect_2 = True
                cam2_kpts2d = ((int((x1+x2)/2), int((y1+y2)/2)))              
        
        ''' ------------ '''
        ''' THIRD CAMERA '''
        ''' ------------ '''
        results = model(frame3)[0]

        if len(results.boxes.data.tolist()) > 0:
            index = 0
            max_score = -1
            max_index = -1
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > max_score:
                    max_score = score
                    max_index = index                    
                
            x1, y1, x2, y2, score, class_id = results.boxes.data.tolist()[max_index]
                    
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            if score > detection_threshold:
                is_detect_3 = True
                cam3_kpts2d = ((int((x1+x2)/2), int((y1+y2)/2)))  
        

        ''' ------------ '''
        ''' FORTH CAMERA '''
        ''' ------------ '''
        results = model(frame4)[0]

        if len(results.boxes.data.tolist()) > 0:
            index = 0
            max_score = -1
            max_index = -1
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > max_score:
                    max_score = score
                    max_index = index                    
                
            x1, y1, x2, y2, score, class_id = results.boxes.data.tolist()[max_index]
                    
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            if score > detection_threshold:
                is_detect_4 = True
                cam4_kpts2d = ((int((x1+x2)/2), int((y1+y2)/2))) 

        # Predict 3D keypoints
        projection_matrices = []
        keypoints = []
        for i in [1, 2, 3, 4]:
            if eval(f"is_detect_{i}"):
                projection_matrices.append(eval(f"P{i}"))
                keypoints.append(eval(f"cam{i}_kpts2d"))
        
        if len(projection_matrices) == 2:
            keypoints_3d = two_DLT(projection_matrices[0], projection_matrices[1], keypoints[0], keypoints[1])
            frame_3ds.append(keypoints_3d)
        elif len(projection_matrices) == 3:
            keypoints_3d = three_DLT(projection_matrices[0], projection_matrices[1], projection_matrices[2], keypoints[0], keypoints[1], keypoints[2])
            frame_3ds.append(keypoints_3d)
        elif len(projection_matrices) == 4:
            keypoints_3d = four_DLT(projection_matrices[0], projection_matrices[1], projection_matrices[2], projection_matrices[3], keypoints[0], keypoints[1], keypoints[2], keypoints[3])
            frame_3ds.append(keypoints_3d)
        else:
            frame_3ds.append([-1, -1, -1])

        if not (ret1 and ret2 and ret3 and ret4):
            break

    x_3d = []
    y_3d = []
    z_3d = []
    for point in frame_3ds:
        x_3d.append(point[0])
        y_3d.append(point[1])
        z_3d.append(point[2])

    # Interpolation
    x_3d = baseball_interpolation(x_3d)
    y_3d = baseball_interpolation(y_3d)
    z_3d = baseball_interpolation(z_3d)
    writeJson(x_3d, y_3d, z_3d)
    




