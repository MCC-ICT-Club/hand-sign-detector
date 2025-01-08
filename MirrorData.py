import os
import cv2 as cv
import time

# Frame Size Variables
imgHeight = 640
imgWidth = 480

gestureLabels = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10']
gestureCount = 0

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
def save_img_locally(frame, class_label, class_dir, img_count):

    img_list = os.listdir(f'./{class_dir}')
    count = 0

    for img in img_list:
        if img[:2] == class_label:
            count += 1
    
    img_count += count

    base_filename = f'{class_label}_{img_count:04d}'
    img_path = get_unique_filename(class_dir, base_filename, '.jpg')
    frame = cv.resize(frame, (imgHeight, imgWidth))
    cv.imwrite(img_path, frame)
    print(f"Image saved: {img_path}")
    img_count = 0
    return img_count

def mirrorImg(classDir):
    pathString = f'/{classDir}'
    imgList = os.listdir(f'./{classDir}')
    writeCount = 0
    count = 0

    # Count Amount of Images in Directory
    for img in imgList:
        if img != "RenameScript.sh":
            print(img)
            writeCount += 1
    
    print(writeCount)

    # Show image
    # for img in imgList:
    #     if img != "RenameScript.sh":
    #         frame = cv.imread(f'./{classDir}/' + img)
    #         cv.imshow("Test", frame)
    #         time.sleep(3)

    # Mirror Images in Directory
    for img in imgList:
        if img != "RenameScript.sh":
            frame = cv.imread(f'./{classDir}/' + img)
            unMirrored = cv.resize(frame, (imgHeight, imgWidth))
            mirrored = cv.flip(unMirrored, 1)
            
            
            # cv.imshow("Test", mirrored)
            # time.sleep(1)
            base_filename = f'{gestureLabels[9]}_{writeCount:04d}'
            imgPath = get_unique_filename(classDir, base_filename, '.jpg')
            print(f"Image Path: {imgPath}")
            # time.sleep(.2)
            cv.imwrite(imgPath, mirrored)
            # print(f"Image: {base_filename} Mirrored!")
            writeCount += 1
            # time.sleep(.2)

def main():
    mirrorImg("labeled/G10")
    

# if file execute standalone then call the main function.
if __name__ == '__main__':
    main()