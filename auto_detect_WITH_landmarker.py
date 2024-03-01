import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

RUN_LANDMARKER = False # Set to False for Gesture Recognition

if RUN_LANDMARKER:
    model_path = '/home/hylander/Documents/RT-Gesture-Detection/models/hand_landmarker.task'
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    HandLandmarkerResult = vision.HandLandmarkerResult
else:
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

if RUN_LANDMARKER:
    # Create a hand landmarker instance with the live stream mode:
    def detector_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=detector_callback)
    detector = HandLandmarker.create_from_options(options)

else:
    # Create a gesture recognizer instance with the live stream mode:
    def recognizer_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        top_gesture = result.gestures
        if np.shape(top_gesture) != (0,):
            # print('\n\nTop Gesture:', top_gesture[0][0].category_name)
            print('\nLandmarks:', result.hand_landmarks[0][0])
            cv2.putText(image, top_gesture[0][0].category_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

    # Convert image color from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Flip the frame horizontally (mirror)
    image = cv2.flip(image, 1)
    # Process image and find hands on mirrored image
    found_hands = hands.process(image)

    # Draw the hand annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if found_hands.multi_hand_landmarks:
        for hand_landmarks in found_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame_timestamp_ms = int(time.time() * 1000)

    if RUN_LANDMARKER:
        # Send live image data to perform hand landmarks detection.
        # The hand landmarker must be created with the live stream mode.
        detector.detect_async(mp_image, frame_timestamp_ms)
    else:
        # Send live image data to perform gesture recognition.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

    # Show the final output
    cv2.imshow("Hand Gesture", image) 