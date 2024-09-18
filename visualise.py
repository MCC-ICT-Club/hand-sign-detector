from tensorflow import keras
from tensorflow.keras.utils import plot_model


def visualize_model(output_image_path):
    # Load the JSON model architecture
    # with open(json_path, 'r') as json_file:
    #     loaded_model_json = json_file.read()
    # loaded_model = keras.models.model_from_json(loaded_model_json)

    # Load the model weights

    model = keras.models.load_model('hand_sign_model.keras', safe_mode=False)

    # Visualize the model architecture and save it to an image file
    plot_model(model, to_file=output_image_path, show_shapes=True, show_layer_names=True)


# Example usage:
output_image_path = 'model_visualization.png'
visualize_model(output_image_path)
