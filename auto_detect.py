import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

model_path = '/home/hylander/Documents/RT-Gesture-Detection/models/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, 
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

## Create a gesture recognizer instance with the live stream mode:

# Callback for recognizer object
def recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print('\n\ngesture recognition result: {}'.format(result))
    top_gesture = result.gestures
    if np.shape(top_gesture) != (0,):
        print('\n\nTop Gesture:', top_gesture[0][0].category_name)
        cv2.putText(image, top_gesture[0][0].category_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        print('\n\n No Gesture Detected!')

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_tracking_confidence=0.3,
    result_callback=recognizer_callback)

recognizer = GestureRecognizer.create_from_options(options)
'''
Note: If you use the live stream mode, you’ll need 
to register a result listener when creating the task. 
The listener is called whenever the task has finished
processing a video frame with the detection result and
the input image as parameters.

2nd Note: If you use the video mode or live stream mode, 
Gesture Recognizer uses tracking to avoid triggering
palm detection model on every frame, and this helps 
to reduce the latency of Gesture Recognizer.
'''

# Start capturing the webcam video
cap = cv2.VideoCapture(0)

global image

while True:
    # If the user presses 'q', exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    # Read the frame from the webcam
    ret, frame = cap.read()
    # Continue if not reading the frame
    if not ret:
        continue
    
    # Get the frame dimensions
    x, y, c = frame.shape

    # Flip the frame horizontally (mirror)
    frame = cv2.flip(frame, 1)
    # Convert image color from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find hands
    found_hands = hands.process(image)

    # Draw the hand annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if found_hands.multi_hand_landmarks:
        for hand_landmarks in found_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame_timestamp_ms = int(time.time() * 1000)

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Send live image data to perform gesture recognition.
    # NOTE: The results are accessible via the `result_callback` provided in
    # the `GestureRecognizerOptions` object.
    # The gesture recognizer must be created with the live stream mode.
    recognizer.recognize_async(mp_image, frame_timestamp_ms)

    # Show the final output
    cv2.imshow("Hand Gesture", image) 