# *****************************************************************************
# ***************************  Python Source Code  ****************************
# *****************************************************************************
#
#   DESIGNER NAME:  Rylan Meilutis and Mario Solis
#
#       FILE NAME:  capture.py
#
# DESCRIPTION
#    This code provides data collection tools for capturing images that can be
#    be used to train the hand_sign.keras CNN model to detect different hand 
#    gestures to apply the model to operate a drone via hand gestures.
#
# *****************************************************************************
import time
import cv2 as cv
import os
import json
import requests
import datetime
import numpy as np
import threading 

cam_device = 0
mirror = True

#  Server Constants
USE_SERVER = False
SERVER_URL = 'http://jupiter:5000'

# Program Setup Constants
MAX_OS  = 3
MIN_OS  = 1

# Data Collection Constants
DRONE_GESTURES = 10

# Camera Setting Variables
img_width = 480
img_height = 640
minBlur = 50
buffer_size = 1
capture_api = [cv.CAP_V4L2, cv.CAP_MSMF]

# Data Capture Controls
key = cv.waitKey(1) & 0xFF
quiteKey = 'q'
captureKey = ' '
singleImage = 's'
videoStream = 'v'
collectTime = 10
imgNum = 0

# Gesture Tracking
gestures = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10']
gestureCount = 0

# Gesture Guide
gestureOne      = "./GestureGuide/GestureOne.jpg"
gestureTwo      = "./GestureGuide/GestureTwo.jpg"
gestureThree    = "./GestureGuide/GestureThree.jpg"
gestureFour     = "./GestureGuide/GestureFour.jpg"
gestureFive     = "./GestureGuide/GestureFive.jpg"
gestureSix      = "./GestureGuide/GestureSix.jpg"
gestureSeven    = "./GestureGuide/GestureSeven.jpg"
gestureEight    = "./GestureGuide/GestureEight.jpg"
gestureNine     = "./GestureGuide/GestureNine.jpg"
gestureTen      = "./GestureGuide/GestureTen.jpg"

# Data Organization Variables
saveImages = True
count = 0
data_dir = 'labeled'  # Directory where captured images will be stored
capture_interval = 0  # Number of frames to skip between captures

# Camera Property Callibration
capture_properties = [
    cv.CAP_PROP_BRIGHTNESS,
    cv.CAP_PROP_CONTRAST,
    cv.CAP_PROP_SATURATION,
    cv.CAP_PROP_GAIN, 
    cv.CAP_PROP_EXPOSURE,
    cv.CAP_PROP_TEMPERATURE,
    cv.CAP_PROP_BACKLIGHT,
]

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def empty(a):
    pass

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def start_message():
    print("Welcome to capture.py\n")
    print("This is the program in charge of gathering data for training")
    print("the hand sign detection model.\n")


# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
# adjust the program for the operating system
def detect_os():
    detected_os = ""
    selected_os = ""
    check_detect = ""
    detect_correct = ""

    # Decision Block - Determines what Operating System is Running
    if os.name == "posix": 
        detected_os = "Linux/Mac"
        selected_os = 0
    elif os.name == "nt":
        detected_os = "Windows"
        selected_os = 1
    while True:
        try:
            check_detect = input(f"I detect that you are using {detected_os}, is this correct [Y/n]: ")
            detect_correct = check_detect.lower()
            if detect_correct != "y" and detect_correct != "n":
                raise ValueError("\nERROR: Must enter either [Y/n]")
            
            if detect_correct == "n":
                print("What operating system are you running this program on?")
                selected_os = int(input("Enter 1 for windows, 2 for mac, or 3 for linux: "))

                if selected_os > MAX_OS or selected_os < MIN_OS:
                    raise ValueError("\nERROR: Must enter a number between 1-3.\n")
                
                if selected_os == 1:
                    selected_os = 1
                else:
                    selected_os = 0
                    
            else:
                selected_os = 0
                    
            break
        except ValueError as e:
            print(e)

    return selected_os

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
# Adjust the program for the hardware of choice
def detect_webcam():
    build_choice = ""
    built_in = ""
    detected_webcam = ""
    while True:
            
        try:
            build_choice = input("Does your system come with a built in webcam [Y/n]: ")
            built_in = build_choice.lower()
            if built_in != "y" and built_in != "n":
                raise ValueError("\nERROR: Must enter either [Y/n]\n")
            if built_in == 'y':
                print("Specify wether you'll be using the built in or external webcam.")
                detect_webcam = int(input("Enter 1 if built in, or enter 2 if external: "))

                if detect_webcam != 1 and detect_webcam != 2:
                    raise ValueError("\nERROR: Must enter either 1 or 2.\n")

                detect_webcam = detect_webcam - 1 
                
            else: 
                print("\nEnsure your external webcam is connected.\n")
                detect_webcam = 0
                
            break
        except ValueError as e:
            print(e)

    return detect_webcam 

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
# Initialized Camera based on user choices
def initialize_camera(cam_num, current_os):
    cam_API = ''

    # Decision Block - Determines what Operating System is Running
    if current_os == 1: 
        cam_API = capture_api[1]
    else:
        cam_API = capture_api[0]

    cap = cv.VideoCapture(cam_num, cam_API)
    if not cap.isOpened():
        retries = 0
        while not cap.isOpened() and retries < 5:
            print("Error: Could not open webcam. Trying again...")
            cap = cv.VideoCapture(cam_device)
            retries += 1
            time.sleep(1)
        print("Error: Could not open webcam.")

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit() 

    retrieve_prior_settings()                       
    cap.set(cv.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def optimize_exposure(cap):

    createLightingWindow(cap)
    
    while True:
        success, frame = cap.read()
        frame_resized = cv.resize(frame, (img_height, img_width))

        cv.putText(frame_resized, f'Adjust exposure to optimal conditions.', (100, 390), 
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv.putText(frame_resized, f'Press q when satisfied.', (100, 400), 
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        cv.imshow("Optimize Exposure", frame_resized)    

        cam_prop_data = open("cam_prop_data.txt", 'w')
        cam_prop_save = open("cam_prop_save.txt", 'w')
        supported_properties = getSupportedProperties(cap)
        for cap_prop in supported_properties:
            value, prop_name = trackbar_cap_prop_values(cap, cap_prop)
            cam_prop, actual_value = setCaptureProperty(cap, cap_prop, value, prop_name)
            cam_prop_data.write(cam_prop)
            cam_prop_save.write(f'{actual_value}\n')

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def gestureGuide(classLabel):
    
    # match classLabel:
    #     case gestures[0]:
    #         image = cv.imread(gestureOne)

    if classLabel ==  gestures[0]:
        image = cv.imread(gestureOne)
    elif classLabel == gestures[1]:
        image = cv.imread(gestureTwo)
    elif classLabel == gestures[2]:
        image = cv.imread(gestureThree)
    elif classLabel == gestures[3]:
        image = cv.imread(gestureFour)
    elif classLabel == gestures[4]:
        image = cv.imread(gestureFive)
    elif classLabel == gestures[5]:
        image = cv.imread(gestureSix)
    elif classLabel == gestures[6]:
        image = cv.imread(gestureSeven)
    elif classLabel == gestures[7]:
        image = cv.imread(gestureEight)
    elif classLabel == gestures[8]:
        image = cv.imread(gestureNine)
    elif classLabel == gestures[9]:
        image = cv.imread(gestureTen)    

    return image

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def setCaptureProperty(cap, prop_id, value, prop_name):
    cap.set(prop_id, value)
    actual_value = cap.get(prop_id)
    camProp = f'Property {prop_id}: \t {prop_name} set to {actual_value}\n'
    return camProp, actual_value

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def trackbar_cap_prop_values(cap, prop_id):
    propName = ""
    match prop_id:
        case 10:
            propName = "Brightness"
            brightness  = cv.getTrackbarPos("Brightness", "Lighting")
            value = brightness
        case 11:
            propName = "Contrast"
            contrast = cv.getTrackbarPos("Contrast", "Lighting")
            value = contrast
        case 12: 
            propName = "Saturation"
            saturation  = cv.getTrackbarPos("Saturation", "Lighting")
            value = saturation
        case 14:
            propName = "Gain"
            gain  = cv.getTrackbarPos("Gain", "Lighting")
            value = gain
        case 15:
            propName = "Exposure"
            exposure    = cv.getTrackbarPos("Exposure", "Lighting")
            value = exposure
        case 23:
            propName = "Temperature"
            temperature = cv.getTrackbarPos("Temperature", "Lighting")
            value = temperature
        case 32:
            propName = "Backlight"
            backlight = cv.getTrackbarPos("Backlight", "Lighting")
            value = 2
        case _:
            propName = "" 
            value = None

    return value, propName

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def getSupportedProperties(cap):
    workingProp = []

    for capProp in capture_properties:
        if cap.get(capProp) != -1.0:
            workingProp.append(capProp)
    
    return workingProp

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------      
def createLightingWindow(cap):
    cv.namedWindow("Lighting")
    cv.resizeWindow("Lighting", 640, 240)
    supported_properties = getSupportedProperties(cap)
    index = 0

    if os.path.exists("cam_prop_save.txt") and open("cam_prop_save.txt", 'r').readline() != "":
        saved_value_list = retrieve_prior_settings()
        for cap_prop in supported_properties:
            value, prop_name = trackbar_cap_prop_values(cap, cap_prop)
            saved_value = saved_value_list[index]
            if cap_prop == 23:
                cv.createTrackbar(prop_name, "Lighting", int(saved_value), 7000, empty)
            elif cap_prop == 15:
                cv.createTrackbar(prop_name, "Lighting", int(saved_value), 1000, empty)
            elif cap_prop == 45:
                cv.createTrackbar(prop_name, "Lighting", int(saved_value), 7000, empty)
            elif cap_prop == 32:
                cv.createTrackbar(prop_name, "Lighting", int(saved_value), 1, empty)
            else:
                cv.createTrackbar(prop_name, "Lighting", int(saved_value), 255, empty)
            index += 1
        index = 0
    else:
        for cap_prop in supported_properties:
            value, prop_name = trackbar_cap_prop_values(cap, cap_prop)
            
            if cap_prop == 23:
                cv.createTrackbar(prop_name, "Lighting", 0, 7000, empty)
            elif cap_prop == 15:
                cv.createTrackbar(prop_name, "Lighting", 0, 1000, empty)
            elif cap_prop == 45:
                cv.createTrackbar(prop_name, "Lighting", 0, 7000, empty)
            elif cap_prop == 32:
                cv.createTrackbar(prop_name, "Lighting", 0, 1, empty)
            else:
                cv.createTrackbar(prop_name, "Lighting", 0, 255, empty)

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def printCapPropSettings(cap):
    for capProp in capture_properties:
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

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def retrieve_prior_settings():
    cam_prop_save = open("cam_prop_save.txt", "r")
    saved_value_list = []
    for cam_prop in capture_properties:
        saved_value = cam_prop_save.readline()
        saved_value = saved_value.rstrip('\n')
        saved_value_list.append(float(saved_value))
    return saved_value_list

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def maintainProperties(cap):
    index = 0
    if os.path.exists("./cam_prop_save.txt"):
        saved_value_list = retrieve_prior_settings()
        for cap_prop in capture_properties:
            saved_value = saved_value_list[index]
            match cap_prop:
                case 10:
                    cap.set(cap_prop, saved_value)
                case 11:
                    cap.set(cap_prop, saved_value)
                case 12:
                    cap.set(cap_prop, saved_value)
                case 14:
                    cap.set(cap_prop, saved_value)
                case 15:
                    cap.set(cap_prop, saved_value)
                case 23:
                    cap.set(cap_prop, saved_value)
                case 32:
                    cap.set(cap_prop, saved_value)
            index += 1
        index = 0

    else:
        for cap_prop in capture_properties:
            match cap_prop:
                case 10:
                    cap.set(cap_prop, 100)
                case 11:
                    cap.set(cap_prop, 44)
                case 12:
                    cap.set(cap_prop, 15)
                case 14:
                    cap.set(cap_prop, 35)
                case 15:
                    cap.set(cap_prop, 600)
                case 23:
                    cap.set(cap_prop, 2200)
                case 32:
                    cap.set(cap_prop, 1)

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def main_menu(cap):
    
    while True:
        # Going to need to explore threading in order to have the window 
        # up while the menu is up at the same time.

        # success, frame = cap.read()
        # frame = cv.resize(frame, (img_height, img_width))
        # cv.imshow("Main Menu", frame)

        try:
            print("Hand Gesture Data Collection: Main Menu")
            print("1. Single Shot Mode")
            print("2. Full Auto Mode")
            print("3. Mode Information")
            print("4. Optomize Exposure")
            print("5. Exit Program")

            user_selection = int(input("Select your mode: "))

            if user_selection > 5 or user_selection < 1:
                raise ValueError("\nERROR: Must enter menu choice between 1-5.\n")
            
            break

        except ValueError as e:
            print(e)
    
    return user_selection

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def mode_selector(selection, cap):
    match selection:
        case 1:
            single_shot_mode(cap)
        case 2:
            full_auto_mode(cap)
        case 3:
            mode_information()
        case 4:
            optimize_exposure(cap)
        case 5:
            exit()

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def mode_information():
    
    cv.destroyAllWindows()

    print("\nMode Breakdown!\n")

    print("Single Shot Mode:")
    print("Lets you collect data one image at a time.")
    print("In single shot mode, you first choose what gesture you want to add images to.")
    print("Once you've specified what gesture you'll be adding images to, place your hand")
    print("in frame with the corresponding gesture held up.")
    print("(refer to the Hand Gesture Map Slides for hand orientation)\n")
    print("Press the space bar to capture pictures of your hand in that specific gesture.")
    print("When satisfied with the images you've collected. You can press q to return")
    print("to the single shot mode menu where you can choose to collect images for a different")
    print("gesture or return to the main menu.\n")

    print("Full Auto Mode:")
    print("Lets you collect data through a video feed.")
    print("In full auto mode you start at gesture 1 and as you collect images")
    print("the mode will move you to the next gesture automatically.")
    print("(refer to the Hand Gesture Map Slides for hand orientation)\n")
    print("With each gesture you choose to add images to the program will save every")
    print("frame that passes over the duration of the countdown. It'll automatically")
    print("move over to the next gesture once done.")
    print("You will be prompted to press 'v' to begin collecting images or ")
    print("you can press 'b' to return to the prior gesture, 'f' to move to the next")
    print("gesture, and 'q' to return to the main menu.\n")

    print("Optomize Exposure:")
    print("This mode give you access to your cameras exposure settings")
    print("For supported webcams this will allow you to bypass its built")
    print("in auto adjust features so you can optimize the camera to your")
    print("current setting.\n")

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def single_shot_mode(cap):
    count = 0
    cv.destroyAllWindows()
    while True:

        # Create the data directory if it doesn't exist
        if not USE_SERVER:
            os.makedirs(data_dir, exist_ok=True)

            # List of class names (labels)
            class_names = get_classes_from_json("classes.json")
        else:
            class_names = get_class_names_from_server()

        try:   
            # Gesture 
            print("Available classes:")
            for idx, class_name in enumerate(class_names):
                print(f"{idx}: {class_name}")
                count += 1
            print(f'Enter: {count} to return to Main Menu')

            # Prompt the user to select a class label
            class_idx = int(input("Enter the index of the class you want to capture images for: "))
            
            if class_idx < 0 and class_idx > count:
                raise ValueError(f"ERROR: Value must be between 0-{count}.")
            
            if class_idx == count:
                print("Returning to Main Menu\n")
                break
            class_label = class_names[class_idx]
            print(f"Capturing images for class: {class_label}")

            # Create a directory for the selected class
            class_dir = os.path.join(data_dir, class_label)
            os.makedirs(class_dir, exist_ok=True)
        except ValueError as e:
            print(e)
            
        frame_count = 0
        img_count = 0

        print("Press 'Spacebar' to capture an image, 'q' to quit.")

        

        while True:
            
            gestureExample = gestureGuide(class_label)
            
            cv.imshow(f"Gesture Examle {class_label}", gestureExample)

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            # Resize the frame if necessary
            frame_resized = cv.resize(frame, (img_height, img_width))
            if mirror:
                frame_resized = cv.flip(frame_resized, 1)

            # Display instructions on the frame
            cv.putText(frame_resized, f'Capturing for class: {class_label}', (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(frame_resized, "Press 'Spacebar' to capture, 'q' to quit", (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame
            cv.imshow('Capture Images', frame_resized)

            key = cv.waitKey(1) & 0xFF

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            # Resize the frame if necessary
            frame_resized = cv.resize(frame, (img_height, img_width))

            if key == ord(' '):  # Spacebar to capture
                if not USE_SERVER:
                    img_count = save_img_locally(frame_resized, class_label, class_dir, img_count)
                else:
                    # Encode the image as a PNG
                    _, img_encoded = cv.imencode('.png', frame_resized)
                    files = {'image': ('image.png', img_encoded.tobytes(), 'image/png')}
                    data = {'class_name': class_label}
                    response = requests.post(SERVER_URL + "/upload", files=files, data=data)
                    if response.status_code == 200:
                        print("Image uploaded and saved successfully.")
                    else:
                        print(f"Failed to upload image. Server responded with status code {response.status_code}")


            elif key == ord('q'):  # 'q' to quit
                print("Quitting...")
                cv.destroyAllWindows()
                count = 0
                break 

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def full_auto_mode(cap):
    gestureCount = 0
    count = 0
    imgNum = 0
    
    cv.destroyAllWindows()

    # Create the data directory if it doesn't exist
    if not USE_SERVER:
        os.makedirs(data_dir, exist_ok=True)

        # List of class names (labels)
        class_names = get_classes_from_json("classes.json")
    else:
        class_names = get_class_names_from_server()

    while True:

        # Live feed
        succes, img = cap.read()
        img = cv.resize(img, (img_height, img_width))
        
        key = cv.waitKey(1) & 0xFF

        cv.putText(img, "Press 'v' to Capture Data " + str(gestures[gestureCount]), (150,440), cv.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255), 1)
        cv.putText(img, "Press 'b' to go back a gesture, or 'f' to go forward ", (120, 400), cv.FONT_HERSHEY_PLAIN, 1, 
                        (255, 0, 255), 1)
       
        cv.imshow("Clear Cam Settings", img)
        maintainProperties(cap)

        if key == ord('b'):
            cv.destroyAllWindows()
            if gestureCount == 0:
                gestureCount = 9
            else:
                gestureCount -= 1

        if key == ord('f'):
            cv.destroyAllWindows()
            if gestureCount == 9:
                gestureCount = 0
            else: 
                gestureCount += 1

        gestureExample = gestureGuide(gestures[gestureCount])

        cv.imshow(f"Gesture Example {gestures[gestureCount]}", gestureExample)
        
        # Initiates Data Collection
        if key == ord(videoStream):
            # Create a directory for the selected class
            class_dir = os.path.join(data_dir, gestures[gestureCount])
            os.makedirs(class_dir, exist_ok=True)

            # Collect Time Logic Variables
            current = datetime.datetime.now()
            newCycle = current + datetime.timedelta(seconds= collectTime)


            # Countdown Logic Variables
            countDown = collectTime
            countCurrent = datetime.datetime.now()
            addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

            while True:
                succes, img_live = cap.read()
                img_live = cv.resize(img_live, (img_height, img_width))

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
                
               

                cv.imshow("Live Feed", img)
                
                success, img = cap.read()
                img = cv.resize(img, (img_height, img_width))

                # Image Capture Logic - Reduces Blury Images
                count += 1
                blur = cv.Laplacian(img, cv.CV_64F).var()
                if count % 1 ==0 and blur < minBlur: 
                    imgNum = save_img_locally(img, gestures[gestureCount], class_dir, imgNum)
                            
                # Capture Logic
                key = cv.waitKey(1) & 0xFF
                if key == ord(captureKey):
                    imgNum = 0
                    break

                # Collect Time Logic Condition
                if datetime.datetime.now() > newCycle: 
                    imgNum = 0
                    gestureCount += 1
                    class_dir = os.path.join(data_dir, gestures[gestureCount])
                    os.makedirs(class_dir, exist_ok=True)
                    break

        if gestureCount == DRONE_GESTURES:
            break
        if key == ord(quiteKey):
            cv.destroyAllWindows()
            break

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def save_img_locally(frame, class_label, class_dir, img_count):

    img_list = os.listdir(f'./{class_dir}')
    count = 0

    for img in img_list:
        if img[:2] == class_label:
            count += 1
    
    img_count += count

    base_filename = f'{class_label}_{img_count:04d}'
    img_path = get_unique_filename(class_dir, base_filename, '.jpg')
    frame = cv.resize(frame, (img_height, img_width))
    cv.imwrite(img_path, frame)
    print(f"Image saved: {img_path}")
    img_count = 0
    return img_count

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------    
def get_class_names_from_server():
    response = requests.get(SERVER_URL + "/classes")
    return response.json()

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# DESCRIPTION
#   This function...
#
# INPUT PARAMETERS:
#   none
# none
#
# OUTPUT PARAMETERS:
#   none
#
# RETURN:
#   none
# -----------------------------------------------------------------------------
def loop(cap):
    while True:
        # succes, img = cap.read()
        # img = cv.resize(img, (img_height, img_width))
        # cv.imshow("Main Menu", img)
        # key = cv.waitKey(1) & 0xFF

        user_choice = main_menu(cap)

        mode_selector(user_choice, cap)
        
        if user_choice == 5:
            break

#---------------------------------------------------------------------
#  main() function
#---------------------------------------------------------------------
def main ():
    #-------------------------------------
    # Variables local to this function
    #-------------------------------------

    start_message()

    try:
        current_os = detect_os()
        selected_cam = detect_webcam()
        cap = initialize_camera(selected_cam, current_os)
        optimize_exposure(cap)
        loop(cap)
     
    except KeyboardInterrupt:
        print()
        print("CTRL-c detected.")
        print()
    finally:
        cap.release()
        cv.destroyAllWindows()
        print("Cam Released and Windows Destroyed")
        print()
        print("******************Program Over*******************")
        print()

# if file execute standalone then call the main function.
if __name__ == '__main__':
  main()