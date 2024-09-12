import os

import cv2
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

tf.get_logger().setLevel('ERROR')
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
labeled_image_dir = 'labeled/'
image_size = (640, 480)  # Resize images to this size
batch_size = 16
epochs = 80
num_classes = 11
use_bounding_boxes = False
output_confusion_matrix_path = 'confusion_matrix.png'  # Path to save confusion matrix

tf.config.run_functions_eagerly(True)


def plot_confusion_matrix(y_true, y_pred, label_dict, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_dict.keys(),
                yticklabels=label_dict.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()


# Helper function to load images and labels
def load_data_from_directory(directory):
    images = []
    labels = []
    for label_name in os.listdir(directory):
        label_path = os.path.join(directory, label_name)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(label_path, img_file)
                    label = label_name
                    if use_bounding_boxes:
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
                    if use_bounding_boxes:
                        for bbox in bounding_boxes:
                            x1, y1, x2, y2 = bbox
                            cropped_img = img[y1:y2, x1:x2]
                            resized_img = cv2.resize(cropped_img, (image_size[1], image_size[0]))
                            images.append(resized_img)
                            labels.append(label)
                    else:
                        resized_img = cv2.resize(img, (image_size[1], image_size[0]))
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
    label_list = sorted(set(labels), key=lambda x: (x[0], int(x[1:])))
    labels_raw = np.array([label_list.index(label) for label in labels])
    labels = np.array([label_dict[label] for label in labels])
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # # Split data into training and validation sets
    split = int(len(images) * 0.8)
    train_images, val_images = images[:split], images[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    #
    # # Create TensorFlow datasets
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    # train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    y_train = labels_raw  # Use labels_raw directly for class weights calculation


    # for images, labels in train_dataset.take(1):
    #     print('Train images shape:', images.shape)
    #     print('Train labels shape:', labels.shape)
    #
    # for images, labels in val_dataset.take(1):
    #     print('Validation images shape:', images.shape)
    #     print('Validation labels shape:', labels.shape)

    # Define the CNN model
    model = Sequential([
        Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Add dropout
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Add dropout
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),  # Add dropout
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model_checkpoint = ModelCheckpoint('hand_sign_model.keras', save_best_only=True)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.5, 1.5],  # Add brightness augmentation
        zoom_range=0.3,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    class_weights = {i: len(y_train) / (len(label_list) * np.bincount(y_train)[i]) for i in range(num_classes)}


    # Train the model
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=epochs,
        class_weight=class_weights,
        validation_data=val_dataset,
        callbacks=[model_checkpoint, early_stopping, lr_scheduler]
    )
    # # Train the model
    # model.fit(
    #     train_dataset,
    #     epochs=epochs,
    #     class_weight = class_weights,
    #     validation_data=val_dataset,
    #     callbacks=[model_checkpoint, early_stopping, lr_scheduler]
    # )

    # Evaluate the model
    loss, accuracy = model.evaluate(val_dataset)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')

    # Predict on validation set
    y_pred = np.argmax(model.predict(val_images), axis=1)
    y_true = np.argmax(val_labels, axis=1)

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred, label_dict, output_confusion_matrix_path)
    print(f'Confusion matrix saved to {output_confusion_matrix_path}')


if __name__ == "__main__":
    main()
