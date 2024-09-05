import os
import shutil

import cv2

# Define paths
raw_image_dir = 'raw/'
labeled_image_dir = 'labeled/'
tmp_dir = "tmp/"
label_names = ['thumbs_up', 'pinkie_out', 'index_up']  # Update with your labels

# Create directories if they don't exist
for label in label_names:
    os.makedirs(os.path.join(labeled_image_dir, label), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir), exist_ok=True)


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, label, img_name
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', img)
        box = (ix, iy, x, y)
        # Save the bounding box to a temporary file
        label_dir = os.path.join(tmp_dir)
        with open(os.path.join(label_dir, img_name + '.txt'), 'a') as f:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")
        print(f"Bounding box saved: {box}")


def label_image(image_path):
    global img, drawing, ix, iy, label, img_name
    img = cv2.imread(image_path)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', draw_rectangle)

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
    images = [f for f in os.listdir(raw_image_dir) if os.path.isfile(os.path.join(raw_image_dir, f))]
    for img_name in images:
        image_path = os.path.join(raw_image_dir, img_name)
        label_image(image_path)


if __name__ == "__main__":
    drawing = False
    ix, iy = -1, -1
    main()
    shutil.rmtree("tmp")
