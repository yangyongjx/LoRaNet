import cv2
import numpy as np
import glob
import pyqtgraph as pg
from numpy.linalg import inv, norm
import sys
from scipy import ndimage, misc
import collections
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import matplotlib.pyplot as plt
pg.mkQApp()



def init_pygame():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL|RESIZABLE)
    gluPerspective(45, (display[0]/display[1]), 0.1, 200)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable( GL_BLEND )
    glTranslatef(0, -15, -100)
    glRotatef(45,-1,0,0)
    # glRotatef(270,0,0,-1)

def surface_param():
    verticies = np.array([
        (-11.5, 11.5, 0), #0
        (11.5, 11.5, 0), #1
        (11.5, 11.5, 8), #2
        (-11.5, 11.5, 8), #3 
        (-11.5, -11.5, 8), #4
        (-11.5, -11.5, 0), #5
        (11.5, -11.5, 0), #6
        (11.5, -11.5, 8)]#7
        )

    print(verticies.shape)
    frustum = np.array([
        (-2, 2, 4),
        (2, 2, 4),
        (0, 0, 0),
        (-2, -2, 4),
        (2, -2, 4)]
        )

    frustum_edge = np.array([
        (0,1,2),
        (0,2,3),
        (3,2,4),
        (1,2,4)]
        )

    edges = np.array([
        (0,1),
        (1,2),
        (2,3),
        (0,3),
        (3,4),
        (4,5),
        (0,5),
        (5,6),
        (4,7),
        (6,7),
        (2,7),
        (1,6)]
        )

    surface = np.array([
        0,1,6,5]
        )
    
    return  verticies, frustum, frustum_edge, edges, surface

def Cube(verticies, edges):
    glBegin(GL_LINES)
    glColor3fv((1,1,1))
    for edge in edges:
        for vertex in edge:
            glVertex3fv(tuple(verticies[vertex]))
    glEnd()

def Face(verticies, surface):
    glBegin(GL_QUADS)
    glColor4fv((0,1,0,0.7))
    for vertex in surface:
        glVertex3fv(tuple(verticies[vertex]))
    glEnd()

def Frustum_cam(frustum_edge, frustum, rt_matrix_inv_1, rt_matrix_inv_2):
    glBegin(GL_TRIANGLES)
    glColor4fv((1,0,1,0.5))
    rt_matrix_inv_2 = np.reshape(rt_matrix_inv_2, -1) / 10

    for frus in frustum_edge:
        for vertex in frus:

            ver = np.matmul(rt_matrix_inv_1, frustum[vertex]) + rt_matrix_inv_2
            glVertex3fv(tuple(ver))
     
    glEnd()

def Frustum_rad(frustum_edge, frustum, r_rad_matrix, t_rad_matrix, rt_matrix_inv_1, rt_matrix_inv_2):
    glBegin(GL_TRIANGLES)
    glColor4fv((1,1,0,0.5))
    t_rad_matrix = np.reshape(t_rad_matrix, -1) / 10
    rt_matrix_inv_2 = np.reshape(rt_matrix_inv_2, -1) / 10


    for frus in frustum_edge:
        for vertex in frus:

            ver = np.matmul(r_rad_matrix, frustum[vertex]) + t_rad_matrix
            ver = np.matmul(rt_matrix_inv_1, ver) + rt_matrix_inv_2
            glVertex3fv(tuple(ver))
     
    glEnd()

def Line():

    glBegin(GL_LINES)
    glColor3f(1,0,0)
    glVertex3f(0,0,0)
    glVertex3f(20,0,0)

    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,20,0)

    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,20)

    glEnd()
    
def cam_config():
    global cap, fps
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('D:/NestData/11-7-2019-64-chirp-16bit/2019-11-07_04-13-53.mp4')
    # cap = cv2.VideoCapture('D:/LogitectVideo/2019-10-05_03-59-14.mp4')
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
  
def calibration():
    global ret, camera_matrix, dist_coeffs, rvecs, tvec, criteria, objp
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25,0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    
    objp = np.zeros((9*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')
    for fname in images:
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners

        ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

        # If found, add object points, image points (after refining them)
        
        if ret == True:
            
            objpoints.append(objp)
            corners2=cv2.cornerSubPix(gray,corners, (9,7), (-1,-1), criteria)
            imgpoints.append(corners)
            
            # Draw and display the corners
            
            cv2.drawChessboardCorners(img, (9,7), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(20)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None,flags =cv2.CALIB_FIX_ASPECT_RATIO)
    rotation_mat = np.zeros(shape=(3, 3))

    np.save("C:/Users/nakorn-vision/Documents/PythonFile/NestProject/camera_matrix", camera_matrix)

    '''
        fine tune focal length
    '''

    camera_matrix[0,0] -= 170
    camera_matrix[1,1] -= 170

    
    # print(camera_matrix, cv2.Rodrigues(rvecs[0], rotation_mat)[0] , gray.shape[::-1])


    cv2.destroyAllWindows()

def draw_function(vec_1_pixel, vec_2_pixel, vec_3_pixel, vec_4_pixel, vec_21_mov_pixel, vec_rad_1_pixel, vec_rad_2_pixel, rt_pixel, vec_rad_0_pixel, vec_0_pixel):

    # cv2.circle(frame, (vec_0_pixel[0], vec_0_pixel[1]), 5, (0,0,255), -1)
    # cv2.circle(frame, (vec_1_pixel[0], vec_1_pixel[1]), 3, (0,0,255), -1)
    # cv2.circle(frame, (vec_2_pixel[0], vec_2_pixel[1]), 3, (255,0,255), -1)
    # cv2.circle(frame, (vec_3_pixel[0], vec_3_pixel[1]), 3, (255,0,0), -1) 

    cv2.circle(frame, (vec_rad_0_pixel[0], vec_rad_0_pixel[1]), 5, (0,0,255), -1)
    cv2.circle(frame, (vec_rad_1_pixel[0], vec_rad_1_pixel[1]), 5, (255,0,0), -1) 
    # cv2.circle(frame, (vec_rad_2_pixel[0], vec_rad_2_pixel[1]), 5, (0,0,255), -1)  
    
    cv2.circle(frame, (rt_pixel[0], rt_pixel[1]), 5, (0,0,255), -1)

    
    '''
        draw vector
    ''' 

    cv2.line(frame, (vec_2_pixel[0], vec_2_pixel[1]), (vec_1_pixel[0], vec_1_pixel[1]), (255,0,0), 2)
    cv2.line(frame, (vec_2_pixel[0] ,vec_2_pixel[1]), (vec_3_pixel[0], vec_3_pixel[1]), (0,0,255), 2)
    cv2.line(frame, (vec_2_pixel[0], vec_2_pixel[1]), (vec_21_mov_pixel[0,0], vec_21_mov_pixel[0,1]), (255,0,255), 2)
    cv2.line(frame, (vec_1_pixel[0], vec_1_pixel[1]), (vec_21_mov_pixel[1,0], vec_21_mov_pixel[1,1]), (255,0,255), 2)
    cv2.line(frame, (vec_3_pixel[0], vec_3_pixel[1]), (vec_21_mov_pixel[2,0], vec_21_mov_pixel[2,1]), (255,0,255), 2)
    cv2.line(frame, (vec_4_pixel[0], vec_4_pixel[1]), (vec_21_mov_pixel[3,0], vec_21_mov_pixel[3,1]), (255,0,255), 2)
    cv2.line(frame ,(vec_21_mov_pixel[0,0], vec_21_mov_pixel[0,1]),(vec_21_mov_pixel[1,0], vec_21_mov_pixel[1,1]),(255,0,0),2)
    cv2.line(frame ,(vec_21_mov_pixel[0,0], vec_21_mov_pixel[0,1]),(vec_21_mov_pixel[2,0], vec_21_mov_pixel[2,1]),(255,0,0),2)
        
def vec_camera_coordinate(rvec, tvec, object_point, u, v):

    r_matrix = cv2.Rodrigues(rvec)[0]
    t_matrix = np.array(tvec)
    
    # print("rcam", r_matrix)
    # rt_matrix = np.hstack((r_matrix,t_matrix))
    
    vec_0 = np.array([[0,0,0]]) ## origin 
    vec_1 = object_point[1] ## vector aruco 20
    vec_2 = object_point[0] ## vector aruco 10
    vec_3 = object_point[2] ## vector aruco 30
    vec_4 = object_point[3] ## vector aruco 40
    
    vec_0_cam = np.matmul(r_matrix, vec_0.T) + t_matrix 
    vec_1_cam = np.matmul(r_matrix, vec_1.T) + t_matrix 
    vec_2_cam = np.matmul(r_matrix, vec_2.T) + t_matrix
    vec_3_cam = np.matmul(r_matrix, vec_3.T) + t_matrix
    vec_4_cam = np.matmul(r_matrix, vec_4.T) + t_matrix
    
    # print("vec_0_cam", vec_0_cam.T)

    vec_12_cam = vec_1_cam - vec_2_cam
    vec_32_cam = vec_3_cam - vec_2_cam
    
    n_vec = np.cross(vec_12_cam.T, vec_32_cam.T)
    p_vec = vec_2_cam.T

    '''
        test point perspective view
    '''
   
    n_vec_uv = np.matmul(camera_matrix, n_vec.T)
    n_vec_uv = n_vec_uv / n_vec_uv[2]
    
    vec_0_pixel = np.matmul(camera_matrix, vec_0_cam)
    vec_1_pixel = np.matmul(camera_matrix, vec_1_cam)
    vec_2_pixel = np.matmul(camera_matrix, vec_2_cam)
    vec_3_pixel = np.matmul(camera_matrix, vec_3_cam)
    vec_4_pixel = np.matmul(camera_matrix, vec_4_cam)


    vec_0_pixel = vec_0_pixel / vec_0_pixel[2]
    vec_1_pixel = vec_1_pixel / vec_1_pixel[2]
    vec_2_pixel = vec_2_pixel / vec_2_pixel[2]
    vec_3_pixel = vec_3_pixel / vec_3_pixel[2]
    vec_4_pixel = vec_4_pixel / vec_4_pixel[2]

    # print(vec_0_pixel)  

    '''
        test z perspective view
    '''
    vec_2_norm = np.linalg.norm(vec_2)
    vec_1_norm = np.linalg.norm(vec_1)

    vec_21_cross = np.cross((vec_2/vec_2_norm), (vec_1/vec_1_norm))
    
    vec_21_mov_2 = vec_21_cross + vec_2
    vec_21_mov_2[0,2] += 70

    vec_21_mov_1 = vec_21_cross + vec_1
    vec_21_mov_1[0,2] += 70

    vec_21_mov_3 = vec_21_cross + vec_3
    vec_21_mov_3[0,2] += 70

    vec_21_mov_4 = vec_21_cross + vec_4
    vec_21_mov_4[0,2] += 70
    

    vec_21_mov = np.array([vec_21_mov_2, vec_21_mov_1, vec_21_mov_3, vec_21_mov_4 ])
    
    vec_21_mov_pixel = []
    
    for i in range(len(vec_21_mov)):

        vec_21_cross_cam = np.matmul(r_matrix, vec_21_mov[i].T) + t_matrix
        vec_21_cross_pixel = np.matmul(camera_matrix, vec_21_cross_cam)
        vec_21_cross_pixel = vec_21_cross_pixel / vec_21_cross_pixel[2]
        vec_21_mov_pixel.append(vec_21_cross_pixel)

    vec_21_mov_pixel = np.array(vec_21_mov_pixel)
    
    '''
        calculate (Rt-p).n = 0
    '''
    if u and v != 0:
        

        fx = camera_matrix[0,0]

        x = u - camera_matrix[0,2]
        y = v - camera_matrix[1,2]
    

        x_p = p_vec[0,0]
        y_p = p_vec[0,1]
        z_p = p_vec[0,2]

        x_pp = n_vec[0,0]
        y_pp = n_vec[0,1]
        z_pp = n_vec[0,2]

        t = ((x_p*x_pp) + (y_p*y_pp) + (z_p*z_pp)) / ((x*x_pp) + (y*y_pp) + (fx*z_pp))
    
        rt = np.array([[t*x], [t*y], [t*fx]])
        rt_pixel = np.matmul(camera_matrix, rt)

        rt_pixel = rt_pixel/rt_pixel[2]

        # print(vec_1_cam.T, vec_2_cam.T, vec_3_cam.T, vec_4_cam.T, "%.4f" % rt[0], "%.4f" % rt[1], "%.4f" % rt[2])
        # print(rt.shape)
        
        # print(vec_1_pixel.T.astype(int), vec_2_pixel.T.astype(int), vec_3_pixel.T.astype(int), vec_4_pixel.T.astype(int))

    else:
        '''
            in case dont have u,v
        '''
        rt_pixel = np.array([[0.5], [0.5], [0.5]])
        
    '''
        inverse transformation matrix
        rt_inverse = [Rotate.t|-Rotate.t*Tran]
        world_coordinate = Rotate.t*cam_coor + (-Rotate.t*Tran)
    '''
    
    rt_matrix_inv_1 = r_matrix.T
    rt_matrix_inv_2 = np.matmul(-1*(r_matrix.T), t_matrix)
    # print("1", rt_matrix_1, "2", rt_matrix_2, "3", r_matrix)
    
   
    return vec_1_pixel, vec_2_pixel, vec_3_pixel, vec_4_pixel, vec_21_mov_pixel, rt_pixel, rt_matrix_inv_1, rt_matrix_inv_2, vec_0_pixel, r_matrix
 
def vec_radar_coordinate(corner_point_radar):

    
    radar_point = np.array([[-19.8,19.8,0], [19.8,19.8,0], [19.8,-19.8,0], [-19.8,-19.8,0]], dtype="float64")
    radar_point = radar_point.reshape(4,1,3)
   
    retval, rvec_r, tvec_r = cv2.solvePnP(radar_point, corner_point_radar, camera_matrix, None)
    
    
    r_rad_matrix = cv2.Rodrigues(rvec_r)[0]
    t_rad_matrix = np.array(tvec_r)

    # print("rradar", r_rad_matrix)

    vec_rad_0 = np.array([[0,0,0]])
    vec_rad_1 = radar_point[0]
    vec_rad_2 = radar_point[1]

    vec_rad_0_cam = np.matmul(r_rad_matrix, vec_rad_0.T) + t_rad_matrix
    vec_rad_1_cam = np.matmul(r_rad_matrix, vec_rad_1.T) + t_rad_matrix 
    vec_rad_2_cam = np.matmul(r_rad_matrix, vec_rad_2.T) + t_rad_matrix 
    
    # 
    # print("vec_rad_0_cam", vec_rad_0_cam.T)

    vec_rad_0_pixel = np.matmul(camera_matrix, vec_rad_0_cam)
    vec_rad_1_pixel = np.matmul(camera_matrix, vec_rad_1_cam)
    vec_rad_2_pixel = np.matmul(camera_matrix, vec_rad_2_cam)

    vec_rad_0_pixel = vec_rad_0_pixel / vec_rad_0_pixel[2]
    vec_rad_1_pixel = vec_rad_1_pixel / vec_rad_1_pixel[2] 
    vec_rad_2_pixel = vec_rad_2_pixel / vec_rad_2_pixel[2]
    
    # print(vec_rad_0_pixel)

    '''
        rvec_r inverse, tvec_r inverse
    '''
    rt_rad_matrix_inv_1 = r_rad_matrix.T
    rt_rad_matrix_inv_2 = np.matmul(-1*(r_rad_matrix.T), t_rad_matrix)


    return vec_rad_1_pixel, vec_rad_2_pixel, vec_rad_0_pixel, r_rad_matrix, t_rad_matrix, rt_rad_matrix_inv_1, rt_rad_matrix_inv_2

def ballRadPosition(rt, rt_rad_matrix_inv_1, rt_rad_matrix_inv_2, param_x, param_y):
    
    global frame_radar
    # print(rt.shape, rt_rad_matrix_inv_1.shape, rt_rad_matrix_inv_2.shape)
    ball_rad_posi = np.matmul(rt_rad_matrix_inv_1, rt) + rt_rad_matrix_inv_2
    
    if (param_x == 0.1) and (param_y == 0.1):
        frame_radar.append(np.array([[0.0, 0.0, 0.0]]))
        # print(np.array([[0.0, 0.0, 0.0]]))
    else:
        frame_radar.append(ball_rad_posi.T)
        # print(ball_rad_posi.T)
        
    
   



def findObject(avg_frame, pixel_aruco_30):

    '''
        detect green color and return coordinate (x1+x, y1+y)
    '''
    # cv2.circle(frame, (int(pixel_aruco_30[0,0]), int(pixel_aruco_30[0,1])), 5, (0,0,255), -1)
    param_x = 0.2
    param_y = 0.2
    px = int(pixel_aruco_30[0,0])
    py = int(pixel_aruco_30[0,1])
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    avg_frame.append(gray_frame)
    np_avg_frame = np.array(avg_frame)
    print(np_avg_frame.shape)
    # print(np_avg_frame.shape[0])
    # print(np_avg_frame[:,100,100])

    if np_avg_frame.shape[0] >= 20:
        sub_frame = cv2.cvtColor(frame[py:py+500, px:px+500, :], cv2.COLOR_BGR2GRAY)
        avg_cell = np_avg_frame[:, py:py+500, px:px+500]
        median_cell = np.median(avg_cell, axis=0)
        median_avg_cell = sub_frame - median_cell
        retval, threshold = cv2.threshold(median_avg_cell, 40, 255, cv2.THRESH_BINARY)
        threshold =  threshold.astype(np.uint8)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c1 = max(contours, key= cv2.contourArea)
            x1,y1,w1,h1 = cv2.boundingRect(c1)
            # print(x1,y1)
            # cv2.circle(frame, (x1,y1), 5, (0,0,255), -1)
            cv2.imshow('median', threshold)
            cv2.imshow('avg', median_avg_cell)
            
            param_x = x1 + px
            param_y = y1 + py

            
    
        elif not contours:
            
            param_x = 0.1
            param_y = 0.1

    # print(param_x, param_y)
        
    return param_x,param_y

# def frameTrigger():

#     frame_green = frame[25, 1125, 1]
#     frame_green = frame_green > 50

#     return frame_green
    



def cam_run():

    
    global frame, frame_count, frame_radar

   
    '''
        save stat
    '''
    # text_file = open('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/frame_number.txt', 'w')
    # rt_text_file = open('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/rt_matrix.txt', 'w')
    # all_frame_detect = open('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/all_frame.txt', 'w')
    '''
        pygame and opengl function
    '''
    ########################################################
    verticies, frustum, frustum_edge, edges, surface = surface_param()
    init_pygame()
    ########################################################

    cam_config()
    ret, frame = cap.read()
    stage_flag = True
    frame_count = 0
    frame_limit = 0
    frame_radar = []

    while ret == True:
   
        '''
            opengl event draw cube and face 
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube(verticies, edges)
        Face(verticies, surface)
        Line()
 
        '''
            aruco detect marker 
        '''
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)
        ids = np.array(ids)
        corners = np.array(corners)
        image_point = []

        '''
            create point in real world coordinate
        '''
        object_point = np.array([[132.5,132.5, 0],[-132.5,132.5,0],[132.5,-132.5,0],[-132.5,-132.5,0]], dtype="float64")
        object_point = object_point.reshape(4,1,3)
        
        '''
            green frame trigger
        '''
        # frame_trigger = frameTrigger()
        frame_trigger = True

        # if frame_trigger:

        if ids.any() and frame_trigger:
            
            # frame_count += 1
            point_constraint = np.array([10,20,30,40,50])
            mask = np.isin(ids, np.array([10,20,30,40,50])) 

            '''
                double check length of ids have to equal to 5
                and mask variable will check the ids which has to be
                10,20,30,40,50
            '''
            if len(ids) == 5 and mask.all() == True :

                point_compare = (ids[:, None] == point_constraint).argmax(axis=0)
                
                for i in range(ids.shape[0]):
                
                    # image_point.append([np.sum(corners[point_compare[0,i],0], axis=0)])
                    image_point.append(corners[point_compare[0,i],0,0])
                
                image_point = np.array(image_point)
                image_point = np.reshape(image_point,(5,1,2))
                # image_point = image_point/4

                '''
                    image point arange re-coordinate to 556 83 64 256 
                    equal to [-11.5,7, 0],[-11.5,-7,0],[11.5,7,0],[11.5,-7,0]
                '''
                image_point_arange = np.array([image_point[0],image_point[1],image_point[2],image_point[3]])
                '''
                    solvpnp and calculate vector of object plane
                '''
                if stage_flag:
                    retval, rvec, tvec = cv2.solvePnP(object_point, image_point_arange, camera_matrix, None)
                    # stage_flag = False

               
                # param_x, param_y = findObject(avg_frame, image_point[2]) 
                
                vec_1_pixel, vec_2_pixel, vec_3_pixel, vec_4_pixel, vec_21_mov_pixel, rt_pixel, rt_matrix_inv_1, rt_matrix_inv_2, \
                vec_0_pixel, r_matrix = vec_camera_coordinate(rvec, tvec, object_point, 0, 0)
               
                '''
                    solvepnp and calculate vector of radar plane
                '''
                vec_rad_1_pixel, vec_rad_2_pixel, vec_rad_0_pixel, r_rad_matrix, t_rad_matrix, rt_rad_matrix_inv_1, rt_rad_matrix_inv_2 = vec_radar_coordinate(np.swapaxes(corners[point_compare[0,4]], 0,1))
                draw_function(vec_1_pixel, vec_2_pixel, vec_3_pixel, vec_4_pixel, vec_21_mov_pixel, vec_rad_1_pixel, vec_rad_2_pixel, rt_pixel, vec_rad_0_pixel, vec_0_pixel)
                
                # frame_limit += 1
                # print('frame_limit', frame_limit)
                # if (param_x != 0.2) and (param_y != 0.2) and (frame_limit <= 500):
                #     ballRadPosition(rt, rt_rad_matrix_inv_1, rt_rad_matrix_inv_2, param_x, param_y)
                #     frame_count += 1
                #     print('frame_collect = ', frame_count)
                    

                '''
                    draw frustum
                '''
                Frustum_cam(frustum_edge, frustum, rt_matrix_inv_1, rt_matrix_inv_2)
                Frustum_rad(frustum_edge, frustum, r_rad_matrix, t_rad_matrix, rt_matrix_inv_1, rt_matrix_inv_2)

            # print(frame_count)
            cv2.imshow('org', frame)
            
        
        
        elif not ids.any():
            cv2.imshow('org', frame)

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        else:
            '''
                enable this function if you want to read from video
            '''
            ########################################################
            # if frame_count > 0 :
            #     # print(frame_count)
            #     text_file.write(str(frame_count) + '\n')
            #     all_frame_detect.write(str(frame_limit) + '\n')
            #     rt_text_file.write(str(r_rad_matrix))
            #     rt_text_file.write(str(t_rad_matrix) + '\n')
            #     np.save("D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/radar_pos_label",np.array(frame_radar))
            #     print('wake text')
            #     print(np.array(frame_radar).shape)

            avg_frame = collections.deque(maxlen=20)    
            frame_count = 0
            frame_limit = 0
            
            ########################################################  

        ret, frame = cap.read()


        pygame.display.flip()

    # text_file.close()
    # rt_text_file.close()
    # all_frame_detect.close()

def main():

    global aruco, dictionary, parameters, camera_matrix, dist_coeffs

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    parameters =  aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 5
    # CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    

    # calibration()
    camera_matrix = np.load("camera_matrix.npy")

    '''
        fine tune focal length
    '''
    camera_matrix[0,0] -= 170
    camera_matrix[1,1] -= 170
    
    print(camera_matrix)
    cam_run()


if __name__ == '__main__':
    main()
   



