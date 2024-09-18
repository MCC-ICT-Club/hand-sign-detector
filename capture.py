import time

import cv2
import os
import json
cam_device = 0
mirror = False

def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
# Configuration
data_dir = 'captured_data'  # Directory where captured images will be stored
img_height = 480
img_width = 640
capture_interval = 1  # Number of frames to skip between captures

# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# List of class names (labels)
class_names = get_classes_from_json("classes.json")

print("Available classes:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")

# Prompt the user to select a class label
class_idx = int(input("Enter the index of the class you want to capture images for: "))
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
        img_filename = f'{class_label}_{img_count:04d}.jpg'
        img_path = os.path.join(class_dir, img_filename)
        frame_resized2 = cv2.resize(frame, (img_width, img_height))
        cv2.imwrite(img_path, frame_resized2)
        print(f"Image saved: {img_path}")
        img_count += 1

    elif key == ord('q'):  # 'q' to quit
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
