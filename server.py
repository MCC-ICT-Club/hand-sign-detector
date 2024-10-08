import os
import re
import time

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import threading
import queue
model_path = 'hand_sign_model_final.keras'

app = Flask(__name__)

model_loaded = False

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
    global model_loaded
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    if model is None:
        print("Failed to load the model.")
        return
    start_time = time.time()
    current_time = time.time()
    model_loaded = True
    print("Model loaded in inference thread.")

    while True:
        try:
            item = request_queue.get(timeout=1)
        except queue.Empty:
            item = None
        if abs(current_time - start_time) > 10:
            model = None
            tf.keras.backend.clear_session()  # Frees up GPU memory
            print("Model unloaded.")
        if item is None:
            break
        if model is None:
            model = tf.keras.models.load_model(model_path)  # Reloads the model
            print("Model reloaded.")
        current_time = time.time()
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

def get_next_file_number(class_name):
    directory = f'uploads/{class_name}'
    if not os.path.exists(directory):
        return 0

    files = os.listdir(directory)
    numbers = []

    for file in files:
        match = re.search(r'(\d+).*$', file)
        if match:
            numbers.append(int(match.group(1)))

    if numbers:
        return max(numbers) + 1
    else:
        return 0

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    # Read the image via file.stream
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image.'}), 400

    class_name = request.form.get('class_name')
    if not class_name or not isinstance(class_name, str) or class_name.strip() == '' or class_name not in label_names:
        return jsonify({'error': 'Class name not provided or not in the current list of classes'}), 400


    # Get the current dimensions of the image
    h, w = image.shape[:2]
    target_w, target_h = image_size  # Assuming image_size is a tuple (width, height)

    # Calculate the aspect ratio for the target and the original image
    target_aspect_ratio = target_w / target_h
    original_aspect_ratio = w / h

    # Cropping the image to maintain the aspect ratio without stretching
    if original_aspect_ratio > target_aspect_ratio:
        # The image is too wide, crop the sides
        new_w = int(target_aspect_ratio * h)
        start_x = (w - new_w) // 2
        cropped_image = image[:, start_x:start_x + new_w]
    else:
        # The image is too tall, crop the top and bottom
        new_h = int(w / target_aspect_ratio)
        start_y = (h - new_h) // 2
        cropped_image = image[start_y:start_y + new_h, :]

    # Resize the cropped image to the target size
    preprocessed_image = cv2.resize(cropped_image, (target_w, target_h))



    num = get_next_file_number(class_name)
    if not os.path.exists(f'uploads/{class_name}'):
        os.makedirs(f'uploads/{class_name}')
    save_path = f"uploads/{class_name}/image_{num}.png"
    if cv2.imwrite(save_path, preprocessed_image):
        return jsonify({'message': 'Image uploaded and saved successfully.'})
    else:
        return jsonify({'error': 'Failed to save the image.'}), 500



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

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found. Please train a model first.'}),

    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet. Please try again later.'}), 500

    h, w = image.shape[:2]
    target_w, target_h = image_size  # Assuming image_size is a tuple (width, height)

    # Calculate the aspect ratio for the target and the original image
    target_aspect_ratio = target_w / target_h
    original_aspect_ratio = w / h

    # Cropping the image to maintain the aspect ratio without stretching
    if original_aspect_ratio > target_aspect_ratio:
        # The image is too wide, crop the sides
        new_w = int(target_aspect_ratio * h)
        start_x = (w - new_w) // 2
        cropped_image = image[:, start_x:start_x + new_w]
    else:
        # The image is too tall, crop the top and bottom
        new_h = int(w / target_aspect_ratio)
        start_y = (h - new_h) // 2
        cropped_image = image[start_y:start_y + new_h, :]

    # Resize the cropped image to the target size
    preprocessed_image = cv2.resize(cropped_image, (target_w, target_h))

    preprocessed_image = preprocess_image(preprocessed_image)
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
    app.run(host="0.0.0.0", port=5000)
