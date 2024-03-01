import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import math

import socket
import sys

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

global unit_pointer 
global image
unit_pointer = (0, 0)

model_path = '/home/hylander/Documents/RT-Gesture-Detection/models/gesture_recognizer.task'
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = vision.RunningMode

# Object to detect hand landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, 
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# -------------------- Creating a server for MATLAB --------------------------
# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind socket to port
server_address = ('localhost', 10000)
print('Starting up on {} port {}'.format(*server_address), file=sys.stderr)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

# ---------------------------------------------------------------------------


## Callback required for livestream mode
def recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Only display results if hand is recognized
    if result.gestures:
        global unit_pointer
        # Display the top gesture
        top_gesture = result.gestures[0][0].category_name
        if str(top_gesture) == 'Closed_Fist':
            print('\n\nTop Gesture:', top_gesture)
            unit_pointer = top_gesture
        # Display the coordinates of the base (mcp, 5) and tip of the index finger (8)
        else:
            index_mcp = result.hand_landmarks[0][5]
            index_tip = result.hand_landmarks[0][8]
            mcp_coord = (index_mcp.x, index_mcp.y)
            tip_coord = (index_tip.x, index_tip.y)

            pointer = (mcp_coord[0] - tip_coord[0], mcp_coord[1] - tip_coord[1])
            unit_pointer = np.round(pointer / np.linalg.norm(pointer) , 2)
            print('\nLandmarks:', unit_pointer)

        cv2.putText(image, top_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

## Create a gesture recognizer instance with the live stream mode:
"""
Note: If you use the live stream mode, you’ll need 
    to register a result listener when creating the task. 
    The listener is called whenever the task has finished
    processing a video frame with the detection result and
    the input image as parameters.
Note 2: If you use the video mode or live stream mode, 
    Gesture Recognizer uses tracking to avoid triggering
    palm detection model on every frame, and this helps 
    to reduce the latency of Gesture Recognizer.
"""
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_tracking_confidence=0.3,
    result_callback=recognizer_callback)
recognizer = GestureRecognizer.create_from_options(options)

# Start capturing the webcam video
cap = cv2.VideoCapture(0)


# TCP Loop to send data 
while True:
    # Wait for a connection
    print('Waiting for a connection', file=sys.stderr)
    connection, client_address = sock.accept()

    try:
        print(f"Connection from {client_address}", file=sys.stderr)        

        while True:
            try:
                # If the user presses 'q', exit the loop
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                # Read the frame from the webcam
                ret, frame = cap.read()
                # Continue if not reading the frame
                if not ret:
                    data = 'No Data'
                    connection.sendall(data.encode())
                    continue


                # Convert image color from BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

                # Mirror the image JUST for display purposes
                image = cv2.flip(image, 1)
                """ NOTE: Image is NOT mirrored when sent to the gesture recognizer.
                    The x,y,z coordinates of landmarks are taken from the top-left of
                    the image. When mirrored, it is taken from the top-right."""

                # Process image and find hands on mirrored image
                found_hands = hands.process(image)
                # Draw the hand annotations on the image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if found_hands.multi_hand_landmarks:
                    for hand_landmarks in found_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                frame_timestamp_ms = int(time.time() * 1000)

                # Send live image data to perform gesture recognition.
                # The gesture recognizer must be created with the live stream mode.
                recognizer.recognize_async(mp_image, frame_timestamp_ms)

                # Show the final output
                cv2.imshow("Hand Gesture", image) 

                ## Send the data over the socket
                data = str(unit_pointer)
                connection.sendall(data.encode()) # 'utf-8'))

                time.sleep(0.01)

            except Exception as e:
                    print(f"Error during data transmission: {e}")
                    break  # Break the inner loop to wait for a new connection
    
    except Exception as e:
        print(f"Error during data transmission: {e}")
        break  # Break the inner loop to wait for a new connection

    finally:
        # Clean up the connection
        print('Closing connection', file=sys.stderr)
        connection.close()
        cap.release()
        cv2.destroyAllWindows()
        break