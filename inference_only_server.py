import time
import cv2
import os
import requests
import concurrent.futures  # For asynchronous server requests

USE_IMAGES = True
SERVER_URL = 'http://jupiter:5000/predict'  # Adjust if your server is running elsewhere
path = "labeled/G2"
cam_device = 0

# Define image size (should match the size used during training)
image_size = (640, 480)

# Initialize variables
last_predicted_class = "Loading..."  # To store the last predicted class
is_waiting_for_response = False  # To track if we're waiting for a server response

def send_frame_to_server(frame):
    # Send the image to the server asynchronously
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(SERVER_URL, files={'image': img_encoded.tobytes()})
    return response


def main():
    global last_predicted_class
    global is_waiting_for_response

    # Open the webcam
    if not USE_IMAGES:
        cap = cv2.VideoCapture(cam_device)
        if not cap.isOpened():
            retries = 0
            while not cap.isOpened() and retries < 5:
                print("Error: Could not open webcam. Trying again...")
                cap = cv2.VideoCapture(cam_device)
                retries += 1
                time.sleep(0.2)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

    print("Press 'q' to quit.")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = None

        run_loop = True
        while run_loop:
            # Capture frame-by-frame
            if not USE_IMAGES:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    continue

                # Preprocess the frame
                # Display the current frame with the last predicted class
                cv2.putText(frame, f'Predicted: {last_predicted_class}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Webcam Feed', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    # If not currently waiting for a server response, send the current frame
                if not is_waiting_for_response:
                    is_waiting_for_response = True  # Mark that we're waiting for a response
                    future = executor.submit(send_frame_to_server, frame)

                # Check if the server response has arrived
                if future and future.done():
                    response = future.result()
                    if response.status_code == 200:
                        result = response.json()
                        last_predicted_class = result['predicted_class']
                    else:
                        print('Error:', response.text)
                        last_predicted_class = 'Error'

                    # Reset the flag to allow sending the next frame
                    is_waiting_for_response = False
            else:
                # Image processing mode if using images instead of webcam
                files = os.listdir(path)
                for i in files:
                    frame = cv2.imread(os.path.join(path, i))

                    response = send_frame_to_server(frame)
                    if response.status_code == 200:
                        result = response.json()
                        last_predicted_class = result['predicted_class']
                    else:
                        print('Error:', response.text)
                        last_predicted_class = 'Error'

                    cv2.putText(frame, f'Predicted: {last_predicted_class}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Image Feed', frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        run_loop = False
                        break

    # Release the webcam and close windows
    if not USE_IMAGES:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
