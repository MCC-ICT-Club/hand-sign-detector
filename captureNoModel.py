import cv2 as cv
import datetime
import numpy as np
import os

# Camera Setting Variables
brightness = 50
camNum = 0
imgWidth = 480
imgHeight = 640
minBlur = 50

# Data Capture Controls
quiteKey = 'q'
captureKey = ' '
singleImage = 's'
videoStream = 'v'
collectTime = 10
imgNum = 0

# Gesture Tracking
gestures = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10']
gestureCount = 0

# Data Organization Variables
imgPath = f'data/images/{gestures[gestureCount]}/'
saveImages = True
count = 0


# Color Detection Variables and Constants
hsvVals = [0, 85, 0, 89, 255, 181]
hsvVals1 = [0, 35, 0, 71, 255, 183]
hsvVals2 = [0, 41, 57, 91, 178, 205]
hsvVals3 = [0, 0, 95, 179, 97, 255]

# Camera Property Callibration
captureProperties = [
    cv.CAP_PROP_BRIGHTNESS,
    cv.CAP_PROP_CONTRAST,
    cv.CAP_PROP_SATURATION,
    cv.CAP_PROP_GAIN, 
    cv.CAP_PROP_EXPOSURE,
    cv.CAP_PROP_TEMPERATURE,
    cv.CAP_PROP_BACKLIGHT,
]
bufferSize = 1


if saveImages:
    os.makedirs(imgPath, exist_ok=True)

def printCapPropSettings():
    for capProp in captureProperties:
        match capProp:
            case 10:
                propName = 'Brightness'
            case 11:
                propName = 'Contrast'
            case 12:
                propName = 'Saturation'
            case 14:
                propName = 'Gain'
            case 15:
                propName = 'Exposure'
            case 23:
                propName = 'Temperature'
            case 32:
                propName = 'Backlight'
        actualValue = cap.get(capProp)

        print(f'Property {capProp}: \t {propName} set to {actualValue}')

def maintainProperties():
    for capProp in captureProperties:
        match capProp:
            case 10:
                cap.set(capProp, 100)
            case 11:
                cap.set(capProp, 44)
            case 12:
                cap.set(capProp, 15)
            case 14:
                cap.set(capProp, 35)
            case 15:
                cap.set(capProp, 600)
            case 23:
                cap.set(capProp, 2200)
            case 32:
                cap.set(capProp, 1)

def initializeCamera(camNum):
    cap = cv.VideoCapture(camNum, cv.CAP_V4L2)
    for capProp in captureProperties:
        match capProp:
            case 10:
                cap.set(capProp, 100)
            case 11:
                cap.set(capProp, 44)
            case 12:
                cap.set(capProp, 15)
            case 14:
                cap.set(capProp, 35)
            case 15:
                cap.set(capProp, 600)
            case 23:
                cap.set(capProp, 2200)
            case 32:
                cap.set(capProp, 1)
    cap.set(cv.CAP_PROP_BUFFERSIZE, bufferSize)
    return cap

def thresholding(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0],hsvVals[1],hsvVals[2]])
    upper = np.array([hsvVals[3],hsvVals[4],hsvVals[5]])
    mask = cv.inRange(hsv, lower, upper)

    lower1 = np.array([hsvVals1[0],hsvVals1[1],hsvVals1[2]])
    upper1 = np.array([hsvVals1[3],hsvVals1[4],hsvVals1[5]])
    mask1 = cv.inRange(hsv, lower1, upper1)

    lower2 = np.array([hsvVals2[0],hsvVals2[1],hsvVals2[2]])
    upper2 = np.array([hsvVals2[3],hsvVals2[4],hsvVals2[5]])
    mask2 = cv.inRange(hsv, lower2, upper2)

    lower3 = np.array([hsvVals3[0],hsvVals3[1],hsvVals3[2]])
    upper3 = np.array([hsvVals3[3],hsvVals3[4],hsvVals3[5]])
    mask3 = cv.inRange(hsv, lower3, upper3)
    

    return mask, mask1, mask2, mask3

def getContours(imgThres, img):
    contours, heirarchy = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #print("Number of contours: ", len(contours))
    boundingRectData = 'BoundingRectSizeData.txt'
    rectData = open(boundingRectData, 'w')
    if contours:
        biggest = max(contours, key=cv.contourArea)
        x,y,w,h = cv.boundingRect(biggest)
        cx = x + w//2
        cy = y + h//2
        rectData.write(f'Height: {h}\n')
        rectData.write(f'Width: {w}\n')
        rectData.write(f'x-coordinate: {x}\n')
        rectData.write(f'y-coordinate: {y}\n')

        # rectData.write("Height: " + str(h) + "\n")
        # rectData.write("Width: " + str(w) + "\n")
        # rectData.write("x-coordinate:  " + str(x) + "\n")
        # rectData.write("y-coordinate: " + str(y) + "\n")
        
        rectData.close()
        # Drawing this Rectangle and printing its Corresponding matrix allows me to begin looking at the shape
        # of the rectangle when whats on screen is just a hand, vs a hand with forearm.
        # By detecting the forearm I can further refine this program to detect when a forearm is present and 
        # give feedback on how to supply a better image.
        rectangle = cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 0)
        #print(rectangle)

        # When rectangle produces a more square box more likely to only be a hand in the image
        # When rectangle produces a more rectangular box more like to be a hand with forearm in the image
        # If theres a hand with forearm in the image then indicate this to the user when adjusting the positioning
        cv.drawContours(img, contours, -1, (255, 0, 255), 7)
        cv.circle(img, (cx, cy), 10, (0,255,0), cv.FILLED)
    else:
        print("No contours detected")
        return None

    return cx

# def findFace(img):
#     faceCascade = cv.CascadeClassifier("./hand-sign-detector/haarcascades/haarcascade_frontalface_default.xml")
#     imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(img, 1.2, 8) # tweaking the second two parameters will give you the ability
#                                                     # to improve the detection capabilities of this method

#     myFaceListC = []
#     myFaceListArea = []

#     for(x,y,w,h) in faces:
#         cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 8)
#         cx = x + w // 2         # Center X
#         cy = y + h // 2         # Center y
#         area = w*h
#         cv.circle(img, (cx,cy), 5, (0, 255, 0), cv.FILLED)
#         myFaceListC.append([cx, cy])
#         myFaceListArea.append(area)

#     if len(myFaceListArea) != 0:
#         i = myFaceListArea.index(max(myFaceListArea))
#         return img, [myFaceListC[i], myFaceListArea[i]]
#     else:
#         return img, [[0,0], 0]

cap = initializeCamera(camNum)
succes, img = cap.read()

while True:
    succes, img = cap.read()
    img = cv.resize(img, (imgHeight, imgWidth))
    cv.imshow("Clear Cam Settings", img)
    key = cv.waitKey(1) & 0xFF

    maintainProperties()
    # printCapPropSettings()

    cv.putText(img, "Press 'v' to Capture Data " + str(gestures[gestureCount]), (150,440), cv.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 255), 1)

    imgThres, imgThres1, imgThres2, imgThres3 = thresholding(img)

    # When looking at all these contours how can I configure the program to focus in on the contour thats most applicable to the current
    # Setting.

    # One approach is in setting it up to detect changes in lighting conditions. Then select the HSV that corresponds to the most optimal lighting 
    # conditions associated with where the program is currenlty running to reduce noise in the codes pick up of contours. 
    # cx = getContours(imgThres, img)
    # cx1 = getContours(imgThres1, img)
    # cx2 = getContours(imgThres2, img)
    cx3 = getContours(imgThres3, img)
    # img, info = findFace(img)

    cv.imshow("Test", img)
    #cv.imshow("Path", imgThres)
    #cv.imshow("Path1", imgThres1)
    #cv.imshow("Path2", imgThres2)
    cv.imshow("Path3", imgThres3)

    # Initiates Data Collection
    if key == ord(videoStream):

        # Collect Time Logic Variables
        current = datetime.datetime.now()
        newCycle = current + datetime.timedelta(seconds = collectTime)

        # Countdown Logic Variables
        countDown = collectTime
        countCurrent = datetime.datetime.now()
        addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

        while True:
            succes, img = cap.read()
            img = cv.resize(img, (imgHeight, imgWidth))

            # Countdown Logic Condition
            if datetime.datetime.now() > addOneSecond:
                countDown -= 1
                countCurrent = datetime.datetime.now()
                addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

            # Displays Capture Status
            cv.putText(img, "Capturing Data, Hold Gesture Until Countdown Ends", (200,400), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)

            # Displays Countdown Logic
            cv.putText(img, "Count Down: " + str(countDown), (350,440), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)
            
            # Display Current Gesture
            cv.putText(img, "Current Gesture: " + str(gestures[gestureCount]), (150,440), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)
            
            cv.imshow("Clear Cam Settings", img)

            # Image Capture Logic - Reduces Blury Images
            succes, img = cap.read()
            count += 1
            blur = cv.Laplacian(img, cv.CV_64F).var()
            # print(f'Blur: {blur}')
            if count % 1 ==0 and blur < minBlur:
                
                imgNum += 1
                img = cv.resize(img, (imgHeight, imgWidth))
                cv.imwrite(imgPath + str(f'{gestures[gestureCount]}_{imgNum:04d}') + ".jpg", img)

            key = cv.waitKey(1) & 0xFF
            if key == ord(captureKey):
                imgNum = 0
                break

            # Collect Time Logic Condition
            if datetime.datetime.now() > newCycle: 
                imgNum = 0
                gestureCount += 1
                imgPath = f'data/images/{gestures[gestureCount]}/'
                if saveImages:
                    os.makedirs(imgPath, exist_ok=True)
                break

    if key == ord(singleImage):
        succes, img = cap.read()
        img = cv.resize(img, (imgHeight, imgWidth))
        cv.imshow("Clear Cam Settings", img)
    if key == ord(quiteKey):
        break

cap.release()
cv.destroyAllWindows()
