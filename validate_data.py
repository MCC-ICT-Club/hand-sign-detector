import cv2
import os
import glob


def show_images_in_folder(folder_path):
    # Get all image files in the folder and its subfolders (jpg, png, jpeg)
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    for ext in image_extensions:
        # Use glob with recursive option (**/*.ext for recursive search)
        image_files.extend(glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True))

    if not image_files:
        print("No images found in the folder.")
        return

    current_index = 0

    while True:
        # Load the current image
        image_path = image_files[current_index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to load image: {image_path}")
            current_index = (current_index + 1) % len(image_files)
            continue

        cv2.imshow('Image Viewer', image)

        key = cv2.waitKey(0)

        if key == 32 or key == 3:  # Space bar to go to next image
            current_index = (current_index + 1) % len(image_files)

        elif key == 8 or key == 127 or key == 40:  # Backspace or delete to delete image
            os.remove(image_path)
            del image_files[current_index]
            if not image_files:
                print("No more images in the folder.")
                break
            current_index = current_index % len(image_files)

        elif key == 81 or key == 2:  # Left arrow to go to the previous image
            current_index = (current_index - 1) % len(image_files)

        elif key == 27:  # Esc to exit
            break

    cv2.destroyAllWindows()


# Usage
folder_path = "validation/"  # Replace with your folder path
show_images_in_folder(folder_path)
