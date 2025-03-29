# USAGE
# python motion_detector.py

import cv2


################################################################
path = 'haarcascades/cascade.xml'  # PATH OF THE CASCADE
cameraNo = 1                       # CAMERA NUMBER
objectName = 'Hemet'       # OBJECT NAME TO DISPLAY
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT
color= (255,0,255)
#################################################################


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

change = 0
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size") #500
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	cap = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
Count =0
# loop over the frames of the video
while True:
## Detecting no. of faces in the frames .
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        smilePath = "haarcascade_smile.xml"
        smileCascade = cv2.CascadeClassifier(smilePath)
        font = cv2.FONT_HERSHEY_SIMPLEX
##        while True:

        ret, image_frame = cap.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image_frame[y:y+h, x:x+w]

            smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.16,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE)

            for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                    cv2.putText(image_frame,'Mask',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
            

            # Save the captured image into the datasets folder
            #cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            # Display the video frame, with bounded rectangle on the person's face
            if (len(faces)) > 1:
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Found more than two persons in ATM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

##
##                    ret, frame = cap.read()
##                    frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
##                    cv2.imwrite('frame.jpg', frame)
##                    print(ocr('frame.jpg'))
                
            cv2.putText(image_frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)
            cv2.imshow('frame', image_frame)

        # To stop taking video, press 'q' for at least 100ms
        #cv2.putText(image_frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)
        if cv2.waitKey(100) & 0xFF == ord('l'):
            break


        cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
        cap.set(10, cameraBrightness)
        # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
        success, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # DETECT THE OBJECT USING THE CASCADE
        scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
        neig=cv2.getTrackbarPos("Neig", "Result")
        objects = cascade.detectMultiScale(gray,scaleVal, neig)
        # DISPLAY THE DETECTED OBJECTS
        for (x,y,w,h) in objects:
                area = w*h
                minArea = cv2.getTrackbarPos("Min Area", "Result")
                if area >minArea:
                    cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
                    cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
                    roi_color = img[y:y+h, x:x+w]

        cv2.imshow("Result", img)
        # grab the current frame and initialize the occupied/unoccupied
        # text
        (grabbed, frame) = cap.read()
        Count = Count+1 
        text = "RefreshFrame"

        if Count > 150:
                firstFrame = None
                Count=0
                

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
                break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
                firstFrame = gray
                continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]  #25

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        # loop over the contours
        for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                        continue
        #	elif(cv2.contourArea(c) > args["min_area"]):
          #              change = change +1
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                count += 1
                (x, y, w, h) = cv2.boundingRect(c)
                #print(contourArea(c))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)  #2
                text = "Occupied"
                #print(cnts) #mine

        #if count>6:
        #        cv2.imshow("activity detected",frame)
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  #0.5
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if cv2.waitKey(1)&0xFF == ord("q"):
                break

        # cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
