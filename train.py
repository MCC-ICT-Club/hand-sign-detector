import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import os

# Define the path to your dataset
data_dir = 'labeled'  # Replace with the path to your 'labeled' folder

# Set image size and batch size
img_height = 480
img_width = 640
batch_size = 32

# Load the dataset with a training and validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,   # 80% training, 20% validation
    subset="training",
    seed=123,               # Seed for reproducibility
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,   # 80% training, 20% validation
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Get class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model architecture
model.summary()

# Define a custom callback to save confusion matrix at each epoch
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, class_names):
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 != 0:  # Change 2 to any interval you prefer
            return
        val_labels = []
        val_predictions = []

        # Iterate over the validation dataset to get predictions
        for images, labels in self.val_data:
            preds = self.model.predict(images, verbose=0)
            val_predictions.extend(np.argmax(preds, axis=1))
            val_labels.extend(labels.numpy())

        # Generate confusion matrix
        cm = confusion_matrix(val_labels, val_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')

        # Save the confusion matrix as a PNG image
        plt.savefig(f'confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()

# Create an instance of the callback
confusion_matrix_callback = ConfusionMatrixCallback(val_ds, class_names)

# Train the model with the custom callback
epochs = 10  # You can adjust the number of epochs
model_checkpoint = ModelCheckpoint('hand_sign_model.keras', save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[confusion_matrix_callback, model_checkpoint]
)

# Generate and save the final confusion matrix after training
val_labels = []
val_predictions = []

for images, labels in val_ds:
    preds = model.predict(images)
    val_predictions.extend(np.argmax(preds, axis=1))
    val_labels.extend(labels.numpy())

# Generate confusion matrix
cm = confusion_matrix(val_labels, val_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix - Final')

# Save the confusion matrix as a PNG image
plt.savefig('confusion_matrix_final.png')
plt.close()
