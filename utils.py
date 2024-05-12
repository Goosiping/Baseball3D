import os
import cv2
import json
import torch
import datetime
import numpy as np

def frameConcatenate(f1, f2, f3, f4, h, w):
    dim = (w, h)
    
    f1 = cv2.resize(f1, dim, interpolation=cv2.INTER_AREA)
    f2 = cv2.resize(f2, dim, interpolation=cv2.INTER_AREA)
    f3 = cv2.resize(f3, dim, interpolation=cv2.INTER_AREA)
    f4 = cv2.resize(f4, dim, interpolation=cv2.INTER_AREA)
    background = cv2.resize(f1, (2*w, 2*h), interpolation=cv2.INTER_AREA)
    background[0:h, 0:w] = f1
    background[0:h, w:2*w] = f2
    background[h:2*h, 0:w] = f3
    background[h:2*h, w:2*w] = f4
    
    return (background)

def getDatetime():
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%Y/%m/%d %H:%M:%S")
    return loc_dt_format

def writeJson(xList, yList, zList):
    filepath = os.path.join('outputs', 'baseball_motion.json')
    data = {}
    data['x'] = xList
    data['y'] = yList
    data['z'] = zList
    
    data['data'] = getDatetime()

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def two_DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return_Point = Vh[3,0:3]/Vh[3,3]
    return_Point[1] = return_Point[1] * -1.0
    
    return return_Point

def three_DLT(P1, P2, P3, point1, point2, point3):
    
        ''' Solve AX=0 '''
        A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:],
            point3[1]*P3[2,:] - P3[1,:],
            P3[0,:] - point3[0]*P3[2,:]
            ]
        A = np.array(A).reshape((6,4))    
    
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
    
        return_Point = Vh[3,0:3]/Vh[3,3]
        return_Point[1] = return_Point[1] * -1.0
        
        return return_Point

def four_DLT(P1, P2, P3, P4, point1, point2, point3, point4):

    ''' Solve AX=0 '''
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:],
         point3[1]*P3[2,:] - P3[1,:],
         P3[0,:] - point3[0]*P3[2,:],
         point4[1]*P4[2,:] - P4[1,:],
         P4[0,:] - point4[0]*P4[2,:]
        ]
    A = np.array(A).reshape((8,4))    

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return_Point = Vh[3,0:3]/Vh[3,3]
    return_Point[1] = return_Point[1] * -1.0
    
    return return_Point

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def read_camera_parameters(camera_id, parameterDirectory):

    inf = open(parameterDirectory + 'cam' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, parameterDirectory):

    inf = open(parameterDirectory + 'rot_trans_cam'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def get_projection_matrix(camera_id, parameterDirectory):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id, parameterDirectory)
    rvec, tvec = read_rotation_translation(camera_id, parameterDirectory)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P