import cv2
import numpy as np


def show_key_codes():
    print("Press any key to see its code. Press ESC to exit.")
    while True:
        # Create a blank image to show a window
        img = 255 * np.ones((100, 400, 3), dtype=np.uint8)
        cv2.imshow('Key Code Finder', img)

        key = cv2.waitKey(0)
        print(f"Key pressed: {key}")

        if key == 27:  # Escape key to exit
            break

    cv2.destroyAllWindows()


# Run this to determine key codes
show_key_codes()
