from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import threading
import queue

app = Flask(__name__)

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
    return normalized_img

# Create a queue for requests
request_queue = queue.Queue()

def inference_thread_func():
    # Load the trained model
    model = tf.keras.models.load_model('hand_sign_model_final.keras')
    print("Model loaded in inference thread.")

    while True:
        item = request_queue.get()
        if item is None:
            break
        input_data, result_queue = item
        # Predict the class
        predictions = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_name = label_names[predicted_class]
        # Put the result in the result_queue
        result_queue.put({'predicted_class': class_name})

# Start the inference thread
inference_thread = threading.Thread(target=inference_thread_func, daemon=True)
inference_thread.start()

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify(get_classes_from_json("classes.json"))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    # Read the image via file.stream
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image.'}), 400

    preprocessed_image = preprocess_image(image)
    # Add batch dimension
    input_data = np.expand_dims(preprocessed_image, axis=0)

    # Create a result queue for this request
    result_queue = queue.Queue()

    # Put the input data and result_queue into the request_queue
    request_queue.put((input_data, result_queue))

    # Wait for the result
    result = result_queue.get()

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
