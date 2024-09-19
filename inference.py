import time

import cv2
import numpy as np
import tensorflow as tf
import json
import os

USE_IMAGES = True
path = "validation/G1"
cam_device = 0


# Load the trained model
model = tf.keras.models.load_model('hand_sign_model.keras')
print("loaded Model")
# Define image size (should match the size used during training)
image_size = (640, 480)



def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Load label names (adjust according to your model's labels)
label_names = get_classes_from_json("classes.json")  # Update with your actual labels


def preprocess_image(image):
    # Resize image to the size expected by the model
    resized_img = cv2.resize(image, [image_size[0], image_size[1]])
    # Normalize the image
    normalized_img = resized_img / 255.0
    # Add batch dimension
    return normalized_img

def main():
    # Open the webcam
    if not USE_IMAGES:
        cap = cv2.VideoCapture(cam_device)
        if not cap.isOpened():
            retries = 0
            while not cap.isOpened() and retries < 5:
                print("Error: Could not open webcam. Trying again...")
                cap = cv2.VideoCapture(cam_device)
                retries += 1
                time.sleep(0.2)
            print("Error: Could not open webcam.")
            return

    print("Press 'q' to quit.")
    run_loop = True
    while run_loop:
        # Capture frame-by-frame
        if not USE_IMAGES:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            preprocessed_frame = preprocess_image(frame)

            # Predict the class
            predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]

            class_name = label_names[predicted_class]

            # Display the result on the frame
            cv2.putText(preprocessed_frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('Webcam Feed', preprocessed_frame)
            key = cv2.waitKey(1) & 0xFF
            # Exit on 'q' key press
            if key == ord('q'):
                break
        else:
            files = os.listdir(path)
            for i in files:
                frame = cv2.imread(os.path.join(path, i))
                preprocessed_frame = preprocess_image(frame)

                # Predict the class
                predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                predicted_class = np.argmax(predictions, axis=1)[0]

                class_name = label_names[predicted_class]
                print(f'Predicted: {class_name}')

                # Display the result on the frame
                cv2.putText(preprocessed_frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow('Webcam Feed', preprocessed_frame)
                key = cv2.waitKey(0) & 0xFF
                # Exit on 'q' key press
                if key == ord('q'):
                    run_loop = False
                    break
                elif key == ord('c'):
                    continue


        # Release the webcam and close windows
    if not USE_IMAGES:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":


    main()
