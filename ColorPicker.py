# *****************************************************************************
# ***************************  Python Source Code  ****************************
# *****************************************************************************
#
#   DESIGNER NAME:  Mario Solis
#
#       FILE NAME:  ColorPicker.py
#
# DESCRIPTION
#    This code provides a means to single out a specific color using the opencv
#    and numpy library. It provides two windows and prints information on the 
#    current HSV value for use in various computer applications where that 
#    value can be applied to detect, read and localize a specific color on 
#    screen. One windows provides a set of trackbars to adjust the HSV values
#    while the other window lets you look at what color is thats being picked up 
#    based on what appears as white in the center image.
#
#    Black out the parts of the image you do not being recognized as the color
#    of interest and make white the parts of the image you do want as the color
#    your looking to detect.
#
# *****************************************************************************

import cv2 as cv
import numpy as np
#import djitellopy as tello

frameWidth = 480
frameHeight = 360

#me = tello.Tello()
#me.connect()
#print(me.get_battery())

camNum = 0
bufferSize = 1
captureProperties = [
    cv.CAP_PROP_AUTO_EXPOSURE, 
    cv.CAP_PROP_ISO_SPEED,
    cv.CAP_PROP_AUTO_WB,
    cv.CAP_PROP_HUE,
    cv.CAP_PROP_GAMMA,
    cv.CAP_PROP_AUTOFOCUS,
    cv.CAP_PROP_EXPOSURE, 
    cv.CAP_PROP_BRIGHTNESS,    
    cv.CAP_PROP_GAIN, 
    cv.CAP_PROP_SATURATION,
    cv.CAP_PROP_TEMPERATURE,
    cv.CAP_PROP_CONTRAST,
    cv.CAP_PROP_WB_TEMPERATURE,
    cv.CAP_PROP_GUID,
    cv.CAP_PROP_IRIS,
    cv.CAP_PROP_BACKLIGHT,
    cv.CAP_PROP_TRIGGER,
    cv.CAP_PROP_MONOCHROME,
    cv.CAP_PROP_FPS,
    cv.CAP_PROP_MODE,
    cv.CAP_PROP_RECTIFICATION,
    cv.CAP_PROP_SHARPNESS,
    cv.CAP_PROP_FOCUS,
    cv.CAP_PROP_SETTINGS,
    cv.CAP_PROP_CODEC_PIXEL_FORMAT,
    cv.CAP_PROP_BITRATE,
    cv.CAP_PROP_FOURCC,
    cv.CAP_PROP_VIDEO_STREAM,
    cv.CAP_PROP_CHANNEL,
]

trackBarNames = []

def empty(a):
    pass

cv.namedWindow("HSV")
cv.resizeWindow("HSV", 640, 240)
cv.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

def initializeCamera(camNum):
    cap = cv.VideoCapture(camNum, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_BUFFERSIZE, bufferSize)
    return cap

def setCaptureProperty(cap, prop_id, value, prop_name):
    cap.set(prop_id, value)
    actual_value = cap.get(prop_id)
    camProp = f'Property {prop_id}: \t {prop_name} set to {actual_value}\n'
    return camProp

def appropriateCapPropValue(cap, prop_id):
    propName = ""
    match prop_id:
        case 21:
            propName = "Auto Exposure"
            value = cap.get(captureProperties[0])
            value = 1
        case 30:
            propName = "ISO Speed"
            value = cap.get(captureProperties[1])
            value = 1
        case 44:
            propName = "Auto WB"
            value = cap.get(captureProperties[2])
            value = 3
        case 13:
            propName = "Hue"
            value = cap.get(captureProperties[3])
            value = 2.0
        case 22:
            propName = "Gamma"
            value = cap.get(captureProperties[4])
            value = 3.0    
        case 39:
            propName = "Autofocus"
            value = cap.get(captureProperties[5])
            value = 3
        case 15:
            propName = "Exposure"
            exposure    = cv.getTrackbarPos("Exposure", "Lighting")
            value = exposure
        case 10:
            propName = "Brightness"
            brightness  = cv.getTrackbarPos("Brightness", "Lighting")
            value = brightness
        case 14:
            propName = "Gain"
            gain  = cv.getTrackbarPos("Gain", "Lighting")
            value = gain
        case 12: 
            propName = "Saturation"
            saturation  = cv.getTrackbarPos("Saturation", "Lighting")
            value = saturation
        case 23:
            propName = "Temperature"
            temperature = cv.getTrackbarPos("Temperature", "Lighting")
            value = temperature
        case 11:
            propName = "Contrast"
            contrast = cv.getTrackbarPos("Contrast", "Lighting")
            value = contrast
        # case 45:
        #     propName = "WB Temperature"
        #     wb_temperature = cv.getTrackbarPos("WB Temperature", "Lighting")
        #     value = wb_temperature
        case 29:
            propName = "GUID"
            guid = cv.getTrackbarPos("GUID", "Lighting")
            value = guid
        case 36:
            propName = "IRIS"
            iris = cv.getTrackbarPos("IRIS", "Lighting")
            value = iris
        case 32:
            propName = "Backlight"
            backlight = cv.getTrackbarPos("Backlight", "Lighting")
            value = 2
        case 24:
            propName = "Trigger"
            trigger = cv.getTrackbarPos("Trigger", "Lighting")
            value = trigger
        case 19:
            propName = "Monochrome"
            monochrome = cv.getTrackbarPos("Monochrome", "Lighting")
            value = monochrome
        case 5:
            propName = "FPS"
            fps = cv.getTrackbarPos("FPS", "Lighting")
            value = fps
        case 9:
            propName = "MODE"
            mode = cv.getTrackbarPos("MODE", "Lighting")
            value = mode
        case 18:
            propName = "Rectification"
            rectification = cv.getTrackbarPos("Rectification", "Lighting")
            value = rectification
        # case 16:
        #     propName = "Convert RGB"
        #     converRGB = cv.getTrackbarPos("Convert RGB", "Lighting")
        #     value = converRGB
        case 20:
            propName = "Sharpness"
            sharpness = cv.getTrackbarPos("Sharpness", "Lighting")
            value = sharpness
        case 28:
            propName = "Focus"
            focus = cv.getTrackbarPos("Focus", "Lighting")
            value = focus
        case 37:
            propName = "Settings"
            settings = cv.getTrackbarPos("Settings", "Lighting")
            value = settings
        case 43:
            propName = "Channel"
            channel = cv.getTrackbarPos("Channel", "Lighting")
            value = channel
        case 46:
            propName = "Code Pixel Format"
            pixelFormat = cv.getTrackbarPos("Code Pixel Format", "Lighting")
            value = pixelFormat
        case 47:
            propName = "Bitrate"
            bitrate = cv.getTrackbarPos("Bitrate", "Lighting")
            value = bitrate
        case 58:
            propName = "Video Stream"
            videoStream = cv.getTrackbarPos("Video Stream", "Lighting")
            value = videoStream
        case 6:
            propName = "FourCC"
            fourCC = cv.getTrackbarPos("FourCC", "Lighting")
            value = fourCC
        case _:
            propName = "" 
            value = None

    return value, propName

def getSupportedProperties():
    workingProp = []

    # for capProp in captureProperties:
    #     if cap.get(capProp) != -1.0 and cap.get(capProp) != 1:
    #         workingProp.append(capProp)

    for capProp in captureProperties:
        if cap.get(capProp) != -1.0:
            workingProp.append(capProp)
    
    return workingProp
        
def createLightingWindow():
    cv.namedWindow("Lighting")
    cv.resizeWindow("Lighting", 640, 240)
    supportedProperties = getSupportedProperties()

    for capProp in supportedProperties:
        value, propName = appropriateCapPropValue(cap, capProp)
        
        if capProp == 23:
            cv.createTrackbar(propName, "Lighting", 0, 7000, empty)
        elif capProp == 15:
            cv.createTrackbar(propName, "Lighting", 0, 1000, empty)
        elif capProp == 45:
            cv.createTrackbar(propName, "Lighting", 0, 7000, empty)
        else:
            cv.createTrackbar(propName, "Lighting", 0, 255, empty)

cap = initializeCamera(camNum)
frameCounter = 0
createLightingWindow()

while True:
    #img = me.get_frame_read().frame
    
    success, img = cap.read()
    img = cv.resize(img, (frameWidth, frameHeight))
    #img = cv.flip(img, 0)
    imgHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("HUE Min", "HSV")
    h_max = cv.getTrackbarPos("HUE Max", "HSV")
    s_min = cv.getTrackbarPos("SAT Min", "HSV")
    s_max = cv.getTrackbarPos("SAT Max", "HSV")
    v_min = cv.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv.getTrackbarPos("VALUE Max", "HSV")

    cap.set(cv.CAP_PROP_CONTRAST, 80)

    camPropData = open("camPropData.txt", 'w')
    supportedProperties = getSupportedProperties()
    for capProp in supportedProperties:
        value, propName = appropriateCapPropValue(cap, capProp)
        camProp = setCaptureProperty(cap, capProp, value, propName)
        camPropData.write(camProp)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask  = cv.inRange(imgHsv, lower, upper)
    result = cv.bitwise_and(img, img, mask=mask)
    # print(f'[{h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max}]')

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv.imshow("Horizontal Stacking", hStack)
    if cv.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
