import cv2 as cv
import os
import numpy as np
import HandTracking as ht

folderPath = "Header"
myList = os.listdir(folderPath)
# list that have all the images matrices
overlayList = []
for imgPath in myList:
    image = cv.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

mylist2 = os.listdir("Header/slider")
sliderList =[]
for imgPath in sliderList:
    image = cv.imread(f'{folderPath}/{imgPath}')
    sliderList.append(image)


# overlayList = [noSelect,white, red, green, sky blue, reset ,eraser]
header = overlayList[0]
# prepare default draw Color
drawColor = (254,254,254)

brushThickness = 12
eraserThickness = 50
xp, yp = 0, 0
frameCanvas = np.zeros((480,640,3),np.uint8)


# Start Capture Video From Webcam Number (0)
webcam = cv.VideoCapture(0)

# Make Hand Tracking Object
detection = ht.handDetector(maxHands=1,minDetectionCon=0.85)

while True:
    # reading frame from webcam
    isTrue , frame = webcam.read()
    # flip frame (mirror effect)
    frame = cv.flip(frame,1)
    # Hand Detection
    detection.findHands(frame,draw=True)
    # Find Hand LandMarks
    landMarks = detection.findPosition(frame,draw=False)
    if len(landMarks) != 0:
        # Tip of index and middle finger
        x1 , y1 = landMarks[8][1:]
        x2 , y2 = landMarks[12][1:]

        # Check which finger are up (1 finger for drawing , 2 fingers for selecting)
        fingers = detection.fingersUp(landMarks)
        # if selection mode (two fingers up - index and middle)
        if fingers[1] and fingers[2]:
            # draw rectangle
            cv.circle(frame, (x1, y1), 15, drawColor, cv.FILLED)
             # cv.rectangle(frame, (x1,y1-25), (x2,y2+25), drawColor,cv.FILLED)
            xp, yp =0,0
            # if we're in the Header:
            if y1 < 78:
                # the range of the first brush
                if 70 < x1 < 150:
                    header = overlayList[1]
                    drawColor = (255,255,255) # white
                elif 160 < x1 < 225:
                    header = overlayList[2]
                    drawColor = (87,87,255) # red
                elif 235 < x1 < 305:
                    header = overlayList[3]
                    drawColor = (119,217,38) # green
                elif 315 < x1 < 375:
                    header = overlayList[4]
                    drawColor = (200,202,29) # sky blue
                elif 463< x1 < 500:
                    header = overlayList[5]
                    # frameCanvas = np.zeros((480,640,3),np.uint8)
                    frameCanvas = frameCanvas * 0
                elif 535 < x1 < 580:
                    header = overlayList[6]
                    drawColor = (0, 0, 0) # eraser
            # cv.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)
            cv.circle(frame, (x1, y1), 15, drawColor, cv.FILLED)

        # if drawing mode (one finger up - index)
        if fingers[1] and fingers[2] == False:
            # draw Circle
            cv.circle(frame,(x1,y1),15,drawColor,cv.FILLED)
            # the first iteration only
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(frameCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else :
                cv.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(frameCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp = x1, y1

    # mixing the two images
    frameGray = cv.cvtColor(frameCanvas,cv.COLOR_BGR2GRAY)
    _, frameInv = cv.threshold(frameGray, 50,255,cv.THRESH_BINARY_INV)
    frameInv = cv.cvtColor(frameInv,cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame,frameInv)
    frame = cv.bitwise_or(frame,frameCanvas)



    frame[0:86,0:640] = header
    # Show FPS
    frame = detection.showFPS(frame,org=(10,460),color=(211, 242, 254))
    # Show the frame
    # mix the canvas with the webcam
    # frame = cv.addWeighted(frame, 0.5,frameCanvas,0.5,0)
    cv.imshow("webcam",frame)
    # Exit on ESC key
    if cv.waitKey(1) & 0xFF == 27:
        break


webcam.release()
cv.destroyAllWindows()
