import numpy as np
import cv2
import sys
import re
import socket
import threading
from turtle import Screen, Turtle
from random import choice



def server_config_1():   
    global client, addr, server
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip = '10.204.226.81'
    port = 9504
    print(ip)
    address = (ip,port)
    server.bind(address)
    server_rec()

def cam_config_1():
    global cap, fps
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPEG'))
    # fourcc = cv2.VideoWriter_fourcc(*"h.264")
    # cap.set(cv2.CAP_OPENCV_MJPEG, 1)
    # cap.set(cv2.CAP_PROP_AUTO_WB,0)
    # cap.set(cv2.CAP_PROP_FOURCC , fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    width = cap.get(3)
    height = cap.get(4)
    print(width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    cam_1()

def cam_1():
    global statusCheck, xyCount, cap, fps
    multiplexim = np.zeros((120,120))
    frameCount = 0
    xyCount = 0
    statusCheck = False
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(fps)
        # Our operations on the frame come here
        # dataPixel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dataPixel = frame
        # dataPixel = dataPixel[5:225,100:320]
        # print(dataPixel)

        # Display the resulting frame

        # if frameCount < 1000 :
        #     multiplexim += dataPixel
        #     # print(type(dataPixel[0,0]))
        # elif frameCount == 1000 :
        #     multiplexim = multiplexim/1000
        #     dataPixel = dataPixel - multiplexim
        #     # print(dataPixel)
        #     # print(multiplexim)
        # elif frameCount > 1000 :
        #     dataPixel = dataPixel - multiplexim
        #     super_thres = dataPixel < 30
        #     dataPixel[super_thres] = 0
        #     # super_thres = dataPixel > 50
        #     # dataPixel[super_thres] = 0
        #     dataPixel = np.round(dataPixel)
        #     dataPixel = dataPixel.astype('uint8')
        #     rr,dataPixel = cv2.threshold(dataPixel,0,255,cv2.THRESH_BINARY_INV)
        #     contours, hierarchy = cv2.findContours(dataPixel,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(dataPixel, contours, -1,(0,255,0), 3)
        #     c  = min(contours, key = cv2.contourArea)
        #     # c = contours[1]
        #     x,y,w,h = cv2.boundingRect(c)
        #     cv2.rectangle(dataPixel,(x,y),(x+w,y+h),(0,0,0),2)
        #     # print(contours)
        #     print(x,y)
            
            
        cv2.rectangle(dataPixel, (350,30) , (910,590), (0,255,0), 2)
        cv2.rectangle(dataPixel, (605,600), (660,655), (255,0 ,0),2)
        cv2.imshow('frame',dataPixel)
        if statusCheck :
            # print(x,y)
            xyCount += 1
            xyMatrix = [x,y]
            xyMatrix = str(xyMatrix)
            with open('/home/ice/PythonFile/impixel.txt','a') as filehandle :
                filehandle.writelines(xyMatrix)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print(frameCount, fps)
        frameCount += 1 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def background_config_1():
    global statusCheck
    colors = ['green', 'red', 'yellow', 'blue']

    screen = Screen()
    screen.bgcolor("red")

    while(True):
		
        if statusCheck == False :
            screen.bgcolor(colors[1])
        elif statusCheck == True :
            screen.bgcolor(colors[0])

    screen.mainloop()


def server_rec():
    global statusCheck
    statusCheck = False
    while True:
        dataIP, addr = server.recvfrom(1024)
        dataIP = dataIP.decode('utf-8')
        print("data: ",dataIP)
        # print(len(dataIP))
        if dataIP == "record_cam\n":
            print("eeeeeee")
            statusCheck = True        
        elif dataIP == "stoprecord_cam\n" :
            print("xxxxxx")
            statusCheck = False
        elif dataIP == "closesocket\n" :
            print("break")
            break

    print('Break------------------------')
    print('number of frame = ', xyCount)
    client.close()
    server.close()

def main():
    # c1 = threading.Thread(name='cam_1',target=cam_config_1)
    ser1 = threading.Thread(name='uwb_1',target=server_config_1)
    bg1 = threading.Thread(name='threading',target=background_config_1)

    # c1.start()
    ser1.start()
    bg1.start()


if __name__ == '__main__':
    main()