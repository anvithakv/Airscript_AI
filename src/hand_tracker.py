"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : hand_tracker.py
Author      : Anvitha K V
Description : Real-time hand tracking using MediaPipe.
              Returns fingertip coordinates and full
              hand landmarks for gesture recognition.
============================================================
"""

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self):
        """Initialize MediaPipe Hands."""

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_draw = mp.solutions.drawing_utils

    def get_hand_data(self, frame):
        """
        Detect the hand and return:

        Returns:
            point       : (x, y) coordinates of index fingertip
            landmarks   : MediaPipe hand landmarks
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            # Index fingertip (Landmark 8)
            fingertip = hand.landmark[8]

            x = int(fingertip.x * w)
            y = int(fingertip.y * h)

            # Draw fingertip
            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 0, 255),
                -1
            )

            return (x, y), hand, handedness

        return None, None, None

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


# ==========================================================
# Testing
# ==========================================================

def main():

    cap = cv2.VideoCapture(0)

    tracker = HandTracker()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        point, landmarks = tracker.get_hand_data(frame)

        if point:

            cv2.putText(
                frame,
                f"Index Tip : {point}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("AirScript AI - Hand Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()