import tensorflow as tf
from beepy import beep
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json

# Define the path to your dataset
data_dir = 'labeled'  # Replace with the path to your 'labeled' folder
val_dir = 'validation'
epochs = 150  # You can adjust the number of epochs

# Set image size and batch size
img_height = 480
img_width = 640
batch_size = 8

# Load the dataset with a training and validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,  # Seed for reproducibility
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Get class names and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load the entire validation dataset into memory
val_images_list = []
val_labels_list = []

for images, labels in val_ds:
    val_images_list.append(images)
    val_labels_list.append(labels)

# Concatenate all images and labels
val_images = tf.concat(val_images_list, axis=0)
val_labels = tf.concat(val_labels_list, axis=0)

# Now, create a new tf.data.Dataset for validation that won't be consumed
val_ds_for_fit = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_ds_for_fit = val_ds_for_fit.batch(batch_size)
val_ds_for_fit = val_ds_for_fit.cache().prefetch(buffer_size=AUTOTUNE)

# Build the CNN model
model = models.Sequential([
    layers.InputLayer(shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Dense(256, activation='relu'),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dense(128, activation='relu'),

    layers.Conv2D(128, (3, 3), activation='relu'), #, kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Dense(128, activation='relu'),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Dense(128, activation='relu'),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model architecture
model.summary()

# Create directory for confusion matrices
os.makedirs('confusion_matrices/epoch_matrices', exist_ok=True)

# Save class names to a JSON file
with open('classes.json', 'w') as f:
    json.dump(class_names, f)


# Define a custom callback to save confusion matrix at each epoch
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_images, val_labels, class_names):
        super().__init__()
        self.val_images = val_images
        self.val_labels = val_labels
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 != 0:  # Change 2 to any interval you prefer
            return

        # Generate predictions
        preds = self.model.predict(self.val_images, verbose=0)
        val_predictions = np.argmax(preds, axis=1)
        val_labels = self.val_labels.numpy()

        # Generate confusion matrix
        cm = confusion_matrix(val_labels, val_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')

        # Save the confusion matrix as a PNG image
        plt.savefig(f'confusion_matrices/epoch_matrices/confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()


# Create an instance of the callback
confusion_matrix_callback = ConfusionMatrixCallback(val_images, val_labels, class_names)

# Train the model with the custom callback
model_checkpoint = ModelCheckpoint('hand_sign_model.keras', save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-7, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds_for_fit,
    epochs=epochs,
    callbacks=[confusion_matrix_callback, model_checkpoint, early_stopping, lr_scheduler]
)

# Generate and save the final confusion matrix after training
preds = model.predict(val_images, verbose=0)
val_predictions = np.argmax(preds, axis=1)
val_labels_array = val_labels.numpy()

# Generate confusion matrix
cm = confusion_matrix(val_labels_array, val_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix - Final')

# Save the confusion matrix as a PNG image
plt.savefig('confusion_matrices/confusion_matrix_final.png')
plt.close()

print("done")
beep(sound="ping")
exit(0)
