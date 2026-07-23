"""
============================================================
                    AirScript AI
------------------------------------------------------------
File        : ui.py
Author      : Anvitha K V
Description : Professional Dashboard UI
============================================================
"""

import cv2
import numpy as np
from config import *

# ==========================================================
# FONT
# ==========================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================================================
# EXTRA COLORS
# ==========================================================

SUCCESS = (0, 220, 0)
WARNING = (0, 200, 255)
ERROR = (0, 0, 255)
INFO = (255, 255, 0)

# ==========================================================
# STATUS COLOR
# ==========================================================

def get_status_color(status):

    status = status.lower()

    if "draw" in status:
        return SUCCESS

    elif "predict" in status:
        return INFO

    elif "wait" in status:
        return WARNING

    elif "hand" in status:
        return ERROR

    return WHITE


# ==========================================================
# DRAW CARD
# ==========================================================

def draw_card(img, x, y, w, h, title=None):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        PANEL,
        -1,
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        BORDER,
        2,
    )

    if title:

        cv2.putText(
            img,
            title,
            (x + 15, y + 30),
            FONT,
            HEADER_FONT,
            WHITE,
            2,
        )

# ==========================================================
# HEADER
# ==========================================================

def draw_header(img):

    # Title
    cv2.putText(
        img,
        "AIRSCRIPT AI",
        (25, 42),
        FONT,
        1.1,
        WHITE,
        3
    )

    # Subtitle
    cv2.putText(
        img,
        "Real-Time Air Writing Recognition",
        (28, 63),
        FONT,
        0.45,
        GRAY,
        1
    )

    # LIVE Indicator
    cv2.circle(
        img,
        (WINDOW_WIDTH - 140, 35),
        7,
        RED,
        -1
    )

    cv2.putText(
        img,
        "LIVE",
        (WINDOW_WIDTH - 125, 41),
        FONT,
        0.55,
        WHITE,
        2
    )

    # Top Divider
    cv2.line(
        img,
        (20, 78),
        (WINDOW_WIDTH - 20, 78),
        BORDER,
        2
    )


# ==========================================================
# CAMERA PANEL
# ==========================================================

def draw_camera_panel(img):

    draw_card(
        img,
        CAMERA_X - 5,
        CAMERA_Y - 5,
        CAMERA_W + 10,
        CAMERA_H + 40,
        "LIVE CAMERA"
    )


# ==========================================================
# CANVAS PANEL
# ==========================================================

def draw_canvas_panel(img):

    draw_card(
        img,
        CANVAS_X - 5,
        CANVAS_Y - 5,
        CANVAS_W + 10,
        CANVAS_H + 40,
        "DRAWING CANVAS"
    )


# ==========================================================
# PLACE CAMERA FRAME
# ==========================================================

def place_camera(img, frame):

    frame = cv2.resize(
        frame,
        (CAMERA_W, CAMERA_H)
    )

    img[
        CAMERA_Y + 30: CAMERA_Y + 30 + CAMERA_H,
        CAMERA_X: CAMERA_X + CAMERA_W
    ] = frame


# ==========================================================
# PLACE DRAWING CANVAS
# ==========================================================

def place_canvas(img, canvas):

    canvas = cv2.resize(
        canvas,
        (CANVAS_W, CANVAS_H)
    )

    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(
            canvas,
            cv2.COLOR_GRAY2BGR
        )

    img[
        CANVAS_Y + 30: CANVAS_Y + 30 + CANVAS_H,
        CANVAS_X: CANVAS_X + CANVAS_W
    ] = canvas

# ==========================================================
# STATUS CARD
# ==========================================================

def draw_status(img, status):

    x = 20
    y = 590
    w = 220
    h = 120

    draw_card(img, x, y, w, h, "STATUS")

    cv2.putText(
        img,
        status,
        (x + 20, y + 72),
        FONT,
        0.65,
        get_status_color(status),
        2,
    )


# ==========================================================
# PREDICTION CARD
# ==========================================================

def draw_prediction(img, prediction):

    x = 260
    y = 590
    w = 180
    h = 120

    draw_card(img, x, y, w, h, "LETTER")

    letter = prediction if prediction else "-"

    size = cv2.getTextSize(
        letter,
        FONT,
        2.2,
        4,
    )[0]

    tx = x + (w - size[0]) // 2

    cv2.putText(
        img,
        letter,
        (tx, y + 82),
        FONT,
        2.2,
        SUCCESS,
        4,
    )


# ==========================================================
# CURRENT WORD CARD
# ==========================================================

def draw_current_word(img, word):

    x = 460
    y = 590
    w = 350
    h = 120

    draw_card(
        img,
        x,
        y,
        w,
        h,
        "CURRENT WORD",
    )

    if not word:
        word = "-"

    cv2.putText(
        img,
        word.upper(),
        (x + 20, y + 75),
        FONT,
        0.9,
        WHITE,
        2,
    )


# ==========================================================
# CONFIDENCE CARD
# ==========================================================

def draw_confidence(img, confidence):

    x = 830
    y = 590
    w = 250
    h = 120

    draw_card(
        img,
        x,
        y,
        w,
        h,
        "CONFIDENCE",
    )

    confidence = max(0, min(confidence, 100))

    # Background Bar
    cv2.rectangle(
        img,
        (x + 20, y + 52),
        (x + 220, y + 72),
        GRAY,
        -1,
    )

    # Filled Bar
    fill = int((confidence / 100) * 200)

    if confidence >= 70:
        bar_color = SUCCESS
    elif confidence >= 40:
        bar_color = WARNING
    else:
        bar_color = ERROR

    cv2.rectangle(
        img,
        (x + 20, y + 52),
        (x + 20 + fill, y + 72),
        bar_color,
        -1,
    )

    cv2.rectangle(
        img,
        (x + 20, y + 52),
        (x + 220, y + 72),
        WHITE,
        1,
    )

    cv2.putText(
        img,
        f"{confidence:.1f}%",
        (x + 70, y + 100),
        FONT,
        0.7,
        WHITE,
        2,
    )

# ==========================================================
# HISTORY PANEL
# ==========================================================

def draw_history(img, history):

    x = 1100
    y = 590
    w = 280
    h = 120

    draw_card(
        img,
        x,
        y,
        w,
        h,
        "HISTORY"
    )

    if history:
        history_text = " → ".join(history[-8:])
    else:
        history_text = "No predictions"

    # Prevent overflow
    if len(history_text) > 35:
        history_text = "..." + history_text[-35:]

    cv2.putText(
        img,
        history_text,
        (x + 15, y + 70),
        FONT,
        0.50,
        WHITE,
        1,
    )


# ==========================================================
# CHARACTER COUNTER
# ==========================================================

def draw_character_counter(img, word):

    text = f"Letters : {len(word)}"

    cv2.putText(
        img,
        text,
        (30, WINDOW_HEIGHT - 15),
        FONT,
        0.55,
        CYAN,
        2,
    )


# ==========================================================
# FOOTER
# ==========================================================

def draw_footer(img):

    footer = "U : Undo      C : Clear      Q : Quit"

    size = cv2.getTextSize(
        footer,
        FONT,
        0.55,
        2,
    )[0]

    x = (WINDOW_WIDTH - size[0]) // 2

    cv2.putText(
        img,
        footer,
        (x, WINDOW_HEIGHT - 15),
        FONT,
        0.55,
        GRAY,
        2,
    )


# ==========================================================
# VERSION
# ==========================================================

def draw_version(img):

    version = "AirScript AI  v1.0"

    size = cv2.getTextSize(
        version,
        FONT,
        0.50,
        1,
    )[0]

    cv2.putText(
        img,
        version,
        (WINDOW_WIDTH - size[0] - 20, WINDOW_HEIGHT - 15),
        FONT,
        0.50,
        GRAY,
        1,
    )

# ==========================================================
# MAIN UI
# ==========================================================

def draw_ui(
    camera_frame,
    canvas,
    status,
    prediction,
    confidence,
    current_word,
    history
):
    """
    Creates the complete AirScript AI dashboard.
    """

    # Background
    dashboard = np.full(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        BACKGROUND,
        dtype=np.uint8
    )

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    draw_header(dashboard)

    # ------------------------------------------------------
    # Camera & Canvas Panels
    # ------------------------------------------------------

    draw_camera_panel(dashboard)
    draw_canvas_panel(dashboard)

    # ------------------------------------------------------
    # Camera
    # ------------------------------------------------------

    place_camera(
        dashboard,
        camera_frame
    )

    # ------------------------------------------------------
    # Canvas
    # ------------------------------------------------------

    place_canvas(
        dashboard,
        canvas
    )

    # ------------------------------------------------------
    # Information Cards
    # ------------------------------------------------------

    draw_status(
        dashboard,
        status
    )

    draw_prediction(
        dashboard,
        prediction
    )

    draw_current_word(
        dashboard,
        current_word
    )

    draw_confidence(
        dashboard,
        confidence
    )

    draw_history(
        dashboard,
        history
    )

    draw_character_counter(
        dashboard,
        current_word
    )

    draw_footer(
        dashboard
    )

    draw_version(
        dashboard
    )

    return dashboard