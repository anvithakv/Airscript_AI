"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : gesture.py
Author      : Anvitha K V
Description : Robust hand gesture recognition
============================================================
"""


class GestureDetector:

    def __init__(self):
        pass

    # ------------------------------------------------------
    # Finger State Detection
    # ------------------------------------------------------

    def get_finger_states(self, hand_landmarks, handedness):

        lm = hand_landmarks.landmark

        # ---------- Thumb ----------

        if handedness == "Right":
            thumb = lm[4].x < lm[3].x
        else:
            thumb = lm[4].x > lm[3].x

        # ---------- Fingers ----------

        index = lm[8].y < lm[6].y
        middle = lm[12].y < lm[10].y
        ring = lm[16].y < lm[14].y
        pinky = lm[20].y < lm[18].y

        return thumb, index, middle, ring, pinky

    # ------------------------------------------------------
    # Gesture Recognition
    # ------------------------------------------------------

    def detect_gesture(self, hand_landmarks, handedness):

        if hand_landmarks is None:
            return "NO_HAND"

        thumb, index, middle, ring, pinky = self.get_finger_states(
            hand_landmarks,
            handedness
        )

        # ☝ Draw
        if (
            not thumb
            and index
            and not middle
            and not ring
            and not pinky
        ):
            return "DRAW"

        # 👍 Predict
        elif (
            thumb
            and not index
            and not middle
            and not ring
            and not pinky
        ):
            return "PREDICT"

        # ✌ Clear
        elif (
            not thumb
            and index
            and middle
            and not ring
            and not pinky
        ):
            return "CLEAR"

        # ✊ Stop
        elif (
            not thumb
            and not index
            and not middle
            and not ring
            and not pinky
        ):
            return "STOP"

        # 🖐 Idle
        elif (
            thumb
            and index
            and middle
            and ring
            and pinky
        ):
            return "IDLE"

        return "UNKNOWN"