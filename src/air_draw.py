"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : air_draw.py
Author      : Anvitha K V
Description : Virtual Air Drawing Canvas
============================================================
"""

import cv2
import numpy as np


class AirDrawer:

    def __init__(self, width=1280, height=720):

        self.width = width
        self.height = height

        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.prev_point = None

        self.line_color = (255, 255, 255)
        self.line_thickness = 8

    # ==========================================================
    # Draw
    # ==========================================================

    def draw(self, point):

        if point is None:
            self.prev_point = None
            return

        if self.prev_point is not None:

            cv2.line(
                self.canvas,
                self.prev_point,
                point,
                self.line_color,
                self.line_thickness,
                cv2.LINE_AA       # Anti-aliased smooth line
            )

        self.prev_point = point

    # ==========================================================
    # Stop Drawing
    # ==========================================================

    def stop_drawing(self):
        """
        Break the current stroke.
        """
        self.prev_point = None

    # ==========================================================
    # Clear Canvas
    # ==========================================================

    def clear(self):

        self.canvas[:] = 0
        self.prev_point = None

    # ==========================================================
    # Get Canvas
    # ==========================================================

    def get_canvas(self):
        return self.canvas