# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:50:12 2023

@author: LLR User
"""


import matplotlib.pyplot as plt
import cv2 
import numpy as np


BLACK = (0,0,0)
WHITE = (255,255,255)
ORANGE = (0,128,255)
RED = (0,0,255)
PURPLE = (128,0,128)
BLUE = (255,0,0)
GREEN= (0,255,0)
YELLOW = (19,235,250)

Lredrange = np.array([0,140,130])
Hredrange = np.array([9,170,200])

Lyellowrange = np.array([28,100,190])
Hyellowrange = np.array([33,170,255])

Lbluerange = np.array([95,100,20])
Hbluerange = np.array([115,255,255])

Lgreenrange = np.array([50,40,100])
Hgreenrange = np.array([65,255,255])

Lorangerange = np.array([12,130,180])
Horangerange = np.array([20,200,210])

Lpurplerange = np.array([130,65,60])
Hpurplerange = np.array([165,180,255])

#upperbody_classifier = cv2.CascadeClassifier(r'C:\Users\LLR User\miniconda3\pkgs\opencv-4.6.0-py311h5d08a89_5\Library\etc\haarcascades\haarcascade_upperbody.xml')
face_classifier = cv2.CascadeClassifier(r'C:\Users\LLR User\miniconda3\pkgs\opencv-4.6.0-py311h5d08a89_5\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
#body_classifier = cv2.CascadeClassifier(r'C:\Users\LLR User\miniconda3\pkgs\opencv-4.6.0-py311h5d08a89_5\Library\etc\haarcascades\haarcascade_fullbody.xml')

class Joint:
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
  
#function to find the body using haarcascade            
# def Find_body(gray):
#      BODY_DATA = body_classifier.detectMultiScale(gray,1.1,1)
#      #for (a,b,c,d) in BODY_DATA:
#          #cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2)    
#      return Joint()
 
# def isolate_area(gray):
#      BODY_DATA = body_classifier.detectMultiScale(gray,1.1,1)
#      #for (a,b,c,d) in BODY_DATA:
#          #cv2.rectangle(frame,(a,b),(a+c,b+d),(0,0,255),2) 
#      return BODY_DATA
   
dimx = 360
dimy = 480

#alternative function to only keep points in the body
# def points_keep(frame,lowerRange,upperRange,bodydata):
#     mask = cv2.inRange(frame, lowerRange, upperRange)
#     with_colour = []
#     if np.sum(bodydata) >0:
#         for row in range(int(bodydata[0,0]),int(bodydata[0,0]+bodydata[0,2]),5):
#             for col in range(int(bodydata[0,1]),int(bodydata[0,1]+bodydata[0,3]),5):
#                 if mask[col,row]==255:
#                     with_colour.append([row,col])
#         return mask, with_colour

#define points to keep, the number 5 makes it look every 5 pixels,
#while not as accurate as doing every pixel it is alot fastter 
def points_keep(frame,lowerRange,upperRange):
    mask = cv2.inRange(frame, lowerRange, upperRange)
    with_colour = []
    for row in range(0,dimx,5):
        for col in range(0,dimy,5):
            if mask[col,row]==255:
                with_colour.append([row,col])
    return mask, with_colour

#Averages coordinated of the mask
def Average_value(masklist):
    average = Average_points_list(masklist)
    joint = Joint(x=average[0],y=average[1])
    return joint
#Haarcascade to find face
def Find_face(gray):
     faces = face_classifier.detectMultiScale(gray)
     for (x,y,w,h) in faces:
         return Joint(x+int(w/2),y+int(h/2))
     return Joint()

#neck or collarbone marker
def Neck_joint(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x+int(w/2),y+3*int(h/2))
    return Joint()
#left shoulder marker
def Neck_left(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x+3*int(w/2),y+4*int(h/2))
    return Joint()
#right shoulder marker    
def Neck_right(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x-int(w/2),y+4*int(h/2))
    return Joint()
#crotch marker
def lower_marker(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x+int(w/2),y+8*int(h/2))
    return Joint()
#right hip marker
def lower_markerR(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x,y+9*int(h/2))
    return Joint()
#left hip marker
def lower_markerL(gray):
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        return Joint(x+2*int(w/2),y+9*int(h/2))
    return Joint()
    
#Find value for colour coordinates
def Average_points_list(list):
    sum_x_coord = 0
    sum_y_coord = 0 
    count = 0
    for item in list:
        sum_x_coord += item[0]
        sum_y_coord += item[1]
        count += 1
    if count >0:
        return [int(sum_x_coord/count),int(sum_y_coord/count)]
    else:
        return [0,0]

#Draw a line between objects classed as joints    
def drawskeleton(jointL, jointR, frame,colour):
    if jointL.x !=0 or jointL.y !=0:
        if jointR.x !=0 or jointR.y !=0:
            cv2.line(frame, (jointL.x, jointL.y), (jointR.x, jointR.y), colour, 1 )


    
#Define function to draw circles around colours it finds

def draw_jointsBLACK(joint,frame):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,BLACK,1)       
def draw_jointsY(joint,frame):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,YELLOW,1)
def draw_jointsB(joint,frame):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,BLUE,1)
def draw_jointsR(joint,frame):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,RED,1)    
def draw_jointsG(joint,frame):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,GREEN,1)
        
def draw_joints(joint,frame,colour):
    if joint.x !=0 or joint.y !=0:
        cv2.circle(frame,(joint.x,joint.y),5,colour,1)
        
vid = cv2.VideoCapture(r'Colour_suit_test.mov')


body_data = np.zeros(4)

while(True):
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read()
    #Resize image, image size is reduced so that process is quicker
    frame = cv2.resize(frame, (dimx,dimy))
    
    """create grey and hsv images"""
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Commented out code detects a body and looks for colours in that
    #Area, is slow and unecessary for results
    #body = Find_body(gray)
    
    # if np.sum(isolate_area(gray))>0:      
    #     body_data = isolate_area(gray)
        
    #     range_image = frame[body_data[0,0]:(body_data[0,0]+body_data[0,2]),
    #                     body_data[0,1]:(body_data[0,1]+body_data[0,3])]
    # else: 
    #     range_image = frame
                
    """Find the head and draw on joints for both head and neck"""

    
    #hsv_inrange = cv2.cvtColor(range_image,cv2.COLOR_BGR2HSV)
    
    """Code to find colour markers"""
    #Convert to HSV
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #to plot just skeleton times make an array of zeros the same size
    #as image and assign and draw joints and connectors
    #frame_skel = gray *0
    
    #find blue marker
    maskBlue, masklistblue = points_keep(frame1,Lbluerange,Hbluerange)
    jointBluemarker = Average_value(masklistblue)
    
    #find red marker
    maskRED, masklistred = points_keep(frame1,Lredrange,Hredrange)
    jointRedmarker = Average_value(masklistred)    
    
    #find green marker
    maskGREEN, masklistgreen = points_keep(frame1,Lgreenrange,Hgreenrange)
    jointgreenmarker = Average_value(masklistgreen)  
    
    #Find purple marker
    maskPURPLE, masklistpurple = points_keep(frame1,Lpurplerange,Hpurplerange)
    jointpurplemarker = Average_value(masklistpurple)
    
    #Find orange marker
    maskORANGE, masklistorange = points_keep(frame1,Lorangerange,Horangerange)
    jointorangemarker = Average_value(masklistorange)
    
    #Find yellow marker
    maskYELLOW, masklistyellow = points_keep(frame1,Lyellowrange,Hyellowrange)
    jointyellowmarker = Average_value(masklistyellow)
    
  
    """Find the head and draw on joints for both head and neck"""

    joint_head = Find_face(gray)
    joint_neck = Neck_joint(gray)
    joint_neckR = Neck_right(gray)
    joint_neckL = Neck_left(gray)
    joint_lower = lower_marker(gray)
    joint_lowerR = lower_markerR(gray)
    joint_lowerL = lower_markerL(gray)
    
    #draw skeleton lines between joints
    drawskeleton(joint_head, joint_neck, frame, RED)
    drawskeleton(joint_neck, joint_neckR, frame, RED)
    drawskeleton(joint_neck, joint_neckL, frame, RED)
    drawskeleton(joint_neck, joint_lower, frame, RED)
    drawskeleton(joint_lowerR, joint_lower, frame, RED)
    drawskeleton(joint_lowerL, joint_lower, frame, RED)
    
    
    #Draw head neck and body joints
    draw_joints(joint_head,frame,WHITE)
    draw_joints(joint_neck,frame,WHITE)
    draw_joints(joint_neckR,frame,WHITE)
    draw_joints(joint_neckL,frame,WHITE)
    draw_joints(joint_lowerR,frame,WHITE)
    draw_joints(joint_lowerL,frame,WHITE)
    

    """code to draw onto colour markers """
    
    #Draw circles on the joints
    draw_joints(jointBluemarker, frame,BLUE)
    drawskeleton(jointBluemarker, joint_lowerL, frame, RED)
    
    draw_joints(jointRedmarker, frame,RED)
    drawskeleton(jointRedmarker, jointgreenmarker, frame, RED)
    
    draw_joints(jointgreenmarker, frame,GREEN)
    drawskeleton(jointgreenmarker, joint_neckR, frame, RED)
    
    draw_joints(jointpurplemarker, frame, PURPLE)
    drawskeleton(jointpurplemarker, joint_lowerR, frame, RED)
    
    draw_joints(jointorangemarker, frame, ORANGE)
    drawskeleton(jointorangemarker, joint_neckL, frame, RED)
    
    draw_joints(jointyellowmarker, frame, YELLOW)
    drawskeleton(jointyellowmarker, jointorangemarker, frame, RED)
    
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    #Commented out code will display the movement of just the thresholded points
    # total = maskBlue+maskGREEN+maskORANGE+maskPURPLE+maskRED+maskYELLOW
    # cv2.imshow('thresholded',total)
    
    
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()

total = maskBlue+maskGREEN+maskORANGE+maskPURPLE+maskRED+maskYELLOW
plt.imshow(total)