import cv2 as cv
import threading

imgHeight = 640
imgWidth = 480
camNum = 0



def initializeCam():
    cap = cv.VideoCapture(camNum, cv.CAP_V4L2)
    return cap

def live_video_stream(cap):
    sucess, img = cap.read()
    resized = cv.resize(img, (imgHeight, imgWidth))
    cv.imshow("Live Feed", resized)

def improvedLiveVideoStream():
    cap = cv.VideoCapture(camNum, cv.CAP_V4L2)
    while True:
        succes, img = cap.read()
        img = cv.resize(img, (imgHeight, imgWidth))
        cv.imshow(f'Live Feed {camNum}', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cap.destroyAllWindows()

liveFeedOne = threading.Thread(target=improvedLiveVideoStream)
liveFeedTwo = threading.Thread(target=improvedLiveVideoStream)

liveFeedOne.start()
liveFeedTwo.start()