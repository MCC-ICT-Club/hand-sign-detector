import cv2 as cv
import datetime
import time
import os

# Camera Setting Variables
brightness = 50
camNum = 0
imgWidth = 480
imgHeight = 640
minBlur = 75

# Data Organization Variables
imgPath = "data/images/"
saveImages = True
count = 0

# Data Capture Controls
quiteKey = 'q'
captureKey = ' '
singleImage = 's'
videoStream = 'v'
collectTime = 5

cap = cv.VideoCapture(camNum)
cap.set(cv.CAP_PROP_BRIGHTNESS, brightness)
succes, img = cap.read()

if saveImages:
    os.makedirs(imgPath, exist_ok=True)

while True:
    succes, img = cap.read()
    cv.imshow("Clear Cam Settings", img)
    key = cv.waitKey(1) & 0xFF

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
            
            # Countdown Logic Condition
            if datetime.datetime.now() > addOneSecond:
                countDown -= 1
                countCurrent = datetime.datetime.now()
                addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

            # Displays Countdown Logic
            cv.putText(img, "Count Down: " + str(countDown), (350,440), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)
            cv.imshow("Clear Cam Settings", img)

            # Image Capture Logic - Reduces Blury Images
            succes, img = cap.read()
            count += 1 
            blur = cv.Laplacian(img, cv.CV_64F).var()
            if count % 4 ==0 and blur > minBlur:
                img = cv.resize(img, (imgHeight, imgWidth))
                cv.imwrite(imgPath + str(f'{count:04d}') + ".png", img)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord(captureKey):
                break
            
            # Collect Time Logic Condition
            if datetime.datetime.now() > newCycle:
                break
        
    if key == ord(singleImage):
        succes, img = cap.read()
        cv.imshow("Clear Cam Settings", img)
    if key == ord(quiteKey):
        break

cap.release()
cv.destroyAllWindows()
