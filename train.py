import os

import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths
labeled_image_dir = 'labeled/'
image_size = (640, 480)  # Resize images to this size
batch_size = 32
epochs = 20
num_classes = 50

tf.config.run_functions_eagerly(True)


# Helper function to load images and labels
def load_data_from_directory(directory):
    images = []
    labels = []
    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(label_path, img_file)
                    label = label_name
                    bounding_boxes = []

                    # Load bounding box file
                    bbox_file = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(bbox_file):
                        with open(bbox_file, 'r') as f:
                            for line in f:
                                bbox = list(map(int, line.strip().split()))
                                bounding_boxes.append(bbox)

                    # Read image
                    img = cv2.imread(img_path)
                    for bbox in bounding_boxes:
                        x1, y1, x2, y2 = bbox
                        cropped_img = img[y1:y2, x1:x2]
                        resized_img = cv2.resize(cropped_img, (image_size[1], image_size[0]))
                        images.append(resized_img)
                        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def main():
    # Load the data
    images, labels = load_data_from_directory(labeled_image_dir)

    # Convert labels to one-hot encoding
    label_dict = {name: idx for idx, name in enumerate(sorted(set(labels)))}
    labels = np.array([label_dict[label] for label in labels])
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # Split data into training and validation sets
    split = int(len(images) * 0.8)
    train_images, val_images = images[:split], images[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for images, labels in train_dataset.take(1):
        print('Train images shape:', images.shape)
        print('Train labels shape:', labels.shape)

    for images, labels in val_dataset.take(1):
        print('Validation images shape:', images.shape)
        print('Validation labels shape:', labels.shape)

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # num_classes for the output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint('hand_sign_model.h5', save_best_only=True)

    # Train the model
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(val_dataset)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
