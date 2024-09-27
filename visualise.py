from tensorflow import keras
from tensorflow.keras.utils import plot_model


def visualize_model(output_image_path):
    model = keras.models.load_model('hand_sign_model_final.keras')

    # Visualize the model architecture and save it to an image file
    plot_model(model, to_file=output_image_path, show_shapes=True, show_layer_names=True)


# Example usage:
output_image_path = 'model_visualization.png'
visualize_model(output_image_path)
