import cv2
import numpy as np
import mediapipe as mp
# import keyboard
# import time
# from tensorflow.keras.models import load_model
from pynput.keyboard import Key, Controller

keyboard = Controller()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, 
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

## Function to get the ULDR direction of the finger from webcam
def get_finger_direction(hand_landmarks, image_height, image_width):
    # Assuming the index finger; landmarks 5, 6, 7, and 8 are relevant
    base_x, base_y = hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y
    tip_x, tip_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

    # Convert from relative to absolute coordinates
    base_x, base_y = int(base_x * image_width), int(base_y * image_height)
    tip_x, tip_y = int(tip_x * image_width), int(tip_y * image_height)

    # Determine direction
    if tip_y < base_y and abs(tip_x - base_x) < abs(tip_y - base_y) / 2:
        return "Up"
    elif tip_y > base_y and abs(tip_x - base_x) < abs(tip_y - base_y) / 2:
        return "Down"
    elif tip_x < base_x and abs(tip_y - base_y) < abs(tip_x - base_x) / 2:
        return "Left"
    elif tip_x > base_x and abs(tip_y - base_y) < abs(tip_x - base_x) / 2:
        return "Right"
    else:
        return "Not Sure"


def is_closed_fist(hand_landmarks, image_height, image_width):
    # Indices for the fingertips of the four fingers excluding the thumb
    fingertip_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                         mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                         mp_hands.HandLandmark.RING_FINGER_TIP,
                         mp_hands.HandLandmark.PINKY_TIP]
    base_indices = [mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                    mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]

    for fingertip, base in zip(fingertip_indices, base_indices):
        fingertip_position = hand_landmarks.landmark[fingertip]
        base_position = hand_landmarks.landmark[base]

        # Check if the fingertip is significantly closer to or below its base in the vertical direction
        if fingertip_position.y < base_position.y:
            return False  # This suggests the finger is not fully bent

    # If the fingertips of the four fingers are not above their bases, consider the fist as closed
    return True




# Start capturing the webcam video
cap = cv2.VideoCapture(0)

use=0

while True:
    # If the user presses 'q', exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    # Get the frame dimensions
    ret, frame = cap.read()
    x, y, c = frame.shape

    # Continue if not reading the frame
    if not ret:
        continue

    # Flip the frame horizontally (mirror)
    frame = cv2.flip(frame, 1)
    # Convert image color from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find hands
    result = hands.process(image)

    # Draw the hand annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_closed_fist(hand_landmarks, y, x):
                gesture = "Stop"
            else:
                gesture = get_finger_direction(hand_landmarks, y, x)

            cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    # Show the final output
    cv2.imshow("Hand Gesture", image) 

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
