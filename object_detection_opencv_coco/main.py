import cv2

threshold=0.5
#load images manually
#img=cv2.imread('lena.PNG')

#Video capture via webcam
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#Step 1
#load the coco data set
classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)

#Step-2
#load the config file mobile-ssd n/w and the weights file.
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'

#Step3
#Setting up ur SSD network takes config path and weights path.
#plus the net set input size,scale(Training purposes params)
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:# infinite video capture by webcam of the sytem.
    success,img=cap.read()
    # Set up Boundry box ,confidence (confs) threshhold and classIds i.e IDs for the object Detected.
    classIds, confs, bbox = net.detect(img,confThreshold=threshold)  # confThreshold means if the network is able to segment an object in iamge about 50 percent then only it creates a bounding box(bbox) i.e classifies it else n/w will ignore it.
    print(classIds,bbox)

    # Step-4
    # traverse through the class id , confidence and the bounding box and make a rectangle around the detected object and use putText to label the object.
    if len(classIds)!=0:#so that if nothing is detected the program do not terminates on its own.
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),bbox):  # flatten to convert matrix to one dimension
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 20, box[1] + 50), cv2.FONT_HERSHEY_DUPLEX, 1,(0, 255, 0), 2)
            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 350, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    # output the labeled image with object.
    cv2.imshow("Output", img)
    cv2.waitKey(0)

