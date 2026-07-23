"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : app.py
Author      : Anvitha K V
Description : Main Application (Dashboard v2 Starter)
============================================================
"""

import cv2
import numpy as np

from hand_tracker import HandTracker
from air_draw import AirDrawer
from predict import predict_canvas
from gesture import GestureDetector
from ui import draw_ui
from writing_mode import WritingMode
from config import *


def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker()
    drawer = AirDrawer()
    gesture_detector = GestureDetector()
    writer = WritingMode()

    prediction = ""
    confidence = 0.0
    status = "Waiting for Hand"

    previous_gesture = ""
    gesture_counter = 0
    GESTURE_THRESHOLD = 10
    prediction_locked = False

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        point, landmarks, handedness = tracker.get_hand_data(frame)

        current_gesture = gesture_detector.detect_gesture(
            landmarks,
            handedness
        )

        if current_gesture == previous_gesture:
            gesture_counter += 1
        else:
            previous_gesture = current_gesture
            gesture_counter = 0
             # Allow the next prediction after the gesture changes
            prediction_locked = False

        if current_gesture == "DRAW":
            status = "Drawing..."
            drawer.draw(point)

        elif current_gesture == "STOP":
            status = "Drawing Stopped"
            drawer.stop_drawing()

        elif (
            current_gesture == "PREDICT"
            and gesture_counter >= GESTURE_THRESHOLD
            and not prediction_locked
        ):

            status = "Prediction Complete"

            prediction, confidence = predict_canvas(drawer.get_canvas())

            if prediction:
                 writer.add_letter(prediction)
                 print("Prediction:", prediction)
                 print("Word after adding:", writer.get_word())
                 drawer.clear()     # Start with a fresh canvas for the next letter

            prediction_locked = True

        elif current_gesture == "CLEAR" and gesture_counter >= GESTURE_THRESHOLD:

            drawer.clear()
            writer.clear()

            prediction = ""
            confidence = 0.0
            status = "Canvas Cleared"

        elif current_gesture == "NO_HAND":
            status = "Waiting for Hand"

        else:
            status = "Recognizing Gesture..."

        dashboard = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), 30, dtype=np.uint8)

        camera = cv2.resize(frame, (640, 480))
        canvas = cv2.resize(drawer.get_canvas(), (640, 480))

        dashboard[70:550, 30:670] = camera
        dashboard[70:550, 730:1370] = canvas

        dashboard = draw_ui(
            frame,
            drawer.get_canvas(),
            status,
            prediction,
            confidence,
            writer.get_word(),
            writer.get_history()
        )



        cv2.imshow("AirScript AI", dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("u"):
            writer.delete_last()

        elif key == ord("q"):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
