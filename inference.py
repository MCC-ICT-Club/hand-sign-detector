import cv2
import numpy as np
import tensorflow as tf
import json

# Load the trained model
model = tf.keras.models.load_model('hand_sign_model.keras')

# Define image size (should match the size used during training)
image_size = (640, 480)
def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['classes']
# Load label names (adjust according to your model's labels)
label_names = get_classes_from_json("classes.json")  # Update with your actual labels


def preprocess_image(image):
    # Resize image to the size expected by the model
    resized_img = cv2.resize(image, image_size)
    # Normalize the image
    normalized_img = resized_img / 255.0
    # Add batch dimension
    batch_img = np.expand_dims(normalized_img, axis=0)
    return batch_img


def main():
    # Open the webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)

        # Predict the class
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_name = label_names[predicted_class]

        # Display the result on the frame
        cv2.putText(frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('Webcam Feed', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
