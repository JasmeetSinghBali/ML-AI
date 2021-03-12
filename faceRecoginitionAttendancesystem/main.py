import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab
#============Attendance via Face Detection and Matching openCV=======================
#Note-> uncomment the lines in the code to activate webcam capturing and commenting the screen capture lines and function.
#3-step process
#1st step to load images, then 2nd to convert the images to rgb and encode the images and 3rd step to capture image from webcam and make it as test image to compare with the already known/encoded images in our attendance directory.
path='image_attendance'
images=[]
classNames=[]
myList=os.listdir(path)
#print(myList)#will give us the image list from our stash

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
#print(images,classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#convert into rgb
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr

encodeListKnown=findEncodings(images)
print('Encoding Completed...')

#capture image from webcam via cv2 method 0 as ID.
#cap=cv2.VideoCapture(0)

#to capture image frame by frame
while True:
    #success,img =cap.read()
    img=captureScreen()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)#appply normalization on the captured image i.e scaling here 0.25 and 0.25 is scaling so that the captured image is smaller and takes less time to process.
    imgS=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#convert webcam image to rgb format

    facesCurFrame=face_recognition.face_locations(imgS)#find face location as their maybe multiple faces in webcam
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)#encode the image on the basis of face location

    #matching and comparing
    for encodeFace,faceLoc in  zip(encodesCurFrame,facesCurFrame):#grabs one face location from facesCurFrame list and grab encoding of encodesCurFrame
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)#already encoded images from attendance directory and the encoded face from webcam image are compared
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)#when we compare the min the faceDis i.e the webcam image matches the best with that attendance directory image
        matchIndex=np.argmin(faceDis)#set match index to the one with the min face distance value among the attendance image directory.

        #create a box and display to which person it matches the most.
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()#find the name of the person via the classnames list we defined earlier by the matchIndex
            print(name)
            y1,x2,y2,x1=faceLoc
            #to tackle the scaling we done earlier so that the rectangle box is in the right place we upscale or rescale by multiplying the face locations by 4.
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

#=====================Basics open cv face detection and matching====================================
#Step-1 loading the images and convert from bgr to rgb (as the package supports rgb image processing.)

#imgElon=face_recognition.load_image_file('imageBasic/elon_musk.jpg')#load image for training
#imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)#convert BGR->RGB
#imgTest=face_recognition.load_image_file('imageBasic/bill_gates.jpg')#test image Loaded
#imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


#Step-2 detect the face in image
#faceLoc= face_recognition.face_locations(imgElon)[0]#since single image so we will get first element of this
#encodeElon = face_recognition.face_encodings(imgElon)[0]

#print(faceLoc) #gives top,right,bottom and left

#cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#faceLocTest= face_recognition.face_locations(imgTest)[0]#since single image so we will get first element of this
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


#Step 3 Compare the two encodings and face detections in previous steps

#results=face_recognition.compare_faces([encodeElon],encodeTest)#by default compare_faces first parameter inputs as list
#faceDis =face_recognition.face_distance([encodeElon],encodeTest)# for best match i.e when we are working with a lot of images practically the lower the distance the better the match is
#print(results,faceDis)# gives a value true if the face of the test and the image trained in our case elon musk matches.


#cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}' ,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#cv2.imshow('Elon Musk',imgElon)
#cv2.imshow('Elon Test',imgTest)
#cv2.waitKey(0)


