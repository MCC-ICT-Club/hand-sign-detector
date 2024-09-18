import os
import shutil
import json

import cv2


def get_classes_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Define paths
raw_image_dir = 'raw/'
labeled_image_dir = 'labeled/'
tmp_dir = "tmp/"
label_names = get_classes_from_json("classes.json")  # Update with your labels
# Create directories if they don't exist
for label in label_names:
    os.makedirs(os.path.join(labeled_image_dir, label), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir), exist_ok=True)


def label_image(image_path):
    global img, drawing, ix, iy, label, img_name
    img = cv2.imread(image_path)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imshow('Image', img)

    print("Press keys to label the image:")
    for i in range(len(label_names)):
        print(f"{i + 1}: {label_names[i]}")
    print("q: Quit")

    while True:
        key = cv2.waitKey(0) & 0xFF

        key_str = chr(key)
        try:
            key_int = int(key_str)
            print(f"Key as integer: {key_int}")
        except ValueError:
            print(f"Key as string: {key_str} (not an integer)")

        if 0 < int(key_str) < len(label_names) + 1:
            label = label_names[int(key_str) - 1]
        elif int(key_str) == 0:
            label = label_names[9]
        elif key_str == "p":
            label = label_names[10]
        elif key == ord('q'):
            break
        else:
            print("Invalid key.")
            continue

        # Save the labeled image and move bounding boxes
        if label:
            labeled_img_path = os.path.join(labeled_image_dir, label, img_name + '.jpg')
            tmp_box_file = os.path.join(tmp_dir, img_name + '.txt')

            # Save image to the labeled directory
            cv2.imwrite(labeled_img_path, img)
            print(f"Labeled image saved as {labeled_img_path}")

            # Move bounding box file if it exists
            if os.path.exists(tmp_box_file):
                shutil.copy(tmp_box_file, labeled_img_path.replace('.jpg', '.txt'))
                os.remove(tmp_box_file)
                print(f"Bounding box file moved to {labeled_img_path.replace('.jpg', '.txt')}")
            break

    cv2.destroyAllWindows()


def main():
    images = [f for f in os.listdir(raw_image_dir) if os.path.isfile(os.path.join(raw_image_dir, f)) and f.endswith(
        '.jpg') or f.endswith('.png')]
    for img_name in images:
        image_path = os.path.join(raw_image_dir, img_name)
        label_image(image_path)


if __name__ == "__main__":
    drawing = False
    ix, iy = -1, -1
    main()
    shutil.rmtree("tmp")
