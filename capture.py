import time
import cv2
import os
import json
import requests
import datetime
import numpy as np

cam_device = 0
mirror = True

USE_SERVER = False
SERVER_URL = 'http://jupiter:5000'

def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
def get_class_names_from_server():
    response = requests.get(SERVER_URL + "/classes")
    return response.json()

def get_unique_filename(directory, base_filename, extension):
    """
    This function returns a unique filename in the given directory by appending a counter if a file already exists.
    """
    counter = 0
    file_path = os.path.join(directory, f"{base_filename}{extension}")
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(directory, f"{base_filename}_{counter}{extension}")
    return file_path

# Configuration
data_dir = 'labeled'  # Directory where captured images will be stored
img_height = 480
img_width = 640
capture_interval = 0  # Number of frames to skip between captures

# Create the data directory if it doesn't exist
if not USE_SERVER:
    os.makedirs(data_dir, exist_ok=True)

# List of class names (labels)
    class_names = get_classes_from_json("classes.json")
else:
    class_names = get_class_names_from_server()

print("Available classes:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")
print("\nEnter 7 for automatic capture mode.\n")

# Prompt the user to select a class label
class_idx = int(input("Enter the index of the class you want to capture images for: "))
if class_idx == 7:
    import cv2 as cv
    
    # Camera Setting Variables
    camNum = 0
    imgWidth = 480
    imgHeight = 640
    minBlur = 50
    bufferSize = 1

    # Data Capture Controls
    quiteKey = 'q'
    captureKey = ' '
    singleImage = 's'
    videoStream = 'v'
    collectTime = 60
    imgNum = 0

    # Gesture Tracking
    gestures = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10']
    gestureCount = 0

    # Data Organization Variables
    imgPath = f'data/images/{gestures[gestureCount]}/'
    saveImages = True
    count = 0

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

    cap = initializeCamera(camNum)
    succes, img = cap.read()

    while True:
        succes, img = cap.read()
        img = cv.resize(img, (imgHeight, imgWidth))

        maintainProperties()

        cv.putText(img, "Press 'v' to Capture Data " + str(gestures[gestureCount]), (150,440), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)

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
                img = cv.resize(img, (imgHeight, imgWidth))

                # Countdown Logic Condition
                if datetime.datetime.now() > addOneSecond:
                    countDown -= 1
                    countCurrent = datetime.datetime.now()
                    addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

                # Displays Capture Status
                cv.putText(img, "Capturing Data, Hold Gesture Until Countdown Ends", (120,400), cv.FONT_HERSHEY_PLAIN, 1,
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
        
        if gestureCount == 10:
            break
        if key == ord(quiteKey):
            break
else:
    class_label = class_names[class_idx]
    print(f"Capturing images for class: {class_label}")

    # Create a directory for the selected class
    class_dir = os.path.join(data_dir, class_label)
    os.makedirs(class_dir, exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(cam_device)
    if not cap.isOpened():
        retries = 0
        while not cap.isOpened() and retries < 5:
            print("Error: Could not open webcam. Trying again...")
            cap = cv2.VideoCapture(cam_device)
            retries += 1
            time.sleep(0.2)
        print("Error: Could not open webcam.")

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    frame_count = 0
    img_count = 0

    print("Press 'Spacebar' to capture an image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Resize the frame if necessary
        frame_resized = cv2.resize(frame, (img_width, img_height))
        if mirror:
            frame_resized = cv2.flip(frame_resized, 1)

        # Display instructions on the frame
        cv2.putText(frame_resized, f'Capturing for class: {class_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, "Press 'Spacebar' to capture, 'q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Capture Images', frame_resized)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar to capture
            if not USE_SERVER:
                base_filename = f'{class_label}_{img_count:04d}'
                img_path = get_unique_filename(class_dir, base_filename, '.jpg')
                frame_resized2 = cv2.resize(frame, (img_width, img_height))
                cv2.imwrite(img_path, frame_resized2)
                print(f"Image saved: {img_path}")
                img_count += 1
            else:
                # Encode the image as a PNG
                _, img_encoded = cv2.imencode('.png', frame_resized)
                files = {'image': ('image.png', img_encoded.tobytes(), 'image/png')}
                data = {'class_name': class_label}
                response = requests.post(SERVER_URL + "/upload", files=files, data=data)
                if response.status_code == 200:
                    print("Image uploaded and saved successfully.")
                else:
                    print(f"Failed to upload image. Server responded with status code {response.status_code}")


        elif key == ord('q'):  # 'q' to quit
            print("Quitting...")
            break

cap.release()
cv2.destroyAllWindows()
