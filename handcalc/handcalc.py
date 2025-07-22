import cv2
import numpy as np
import mediapipe as mp
import math
import time

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# --- Rainbow Colors ---
RAINBOW_COLORS = [
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255),  # Magenta (Indigo/Violet substitute)
]
color_index = 0

# --- Button class with better UI and round shape ---
class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.text = text
        self.size = size
        self.radius = size[0] // 2  # For round buttons

    def draw(self, img, color=(100, 100, 100), text_color=(255, 255, 255)): # Darker grey for buttons, white text
        x, y = self.pos
        w, h = self.size
        center = (x + self.radius, y + self.radius)
        
        # Draw circular button
        cv2.circle(img, center, self.radius, color, cv2.FILLED)
        cv2.circle(img, center, self.radius, (150, 150, 150), 2) # Lighter grey border

        # Put text in the center
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2 - 5 # Adjust for vertical centering
        cv2.putText(img, self.text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

    def is_pressed(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        # For a circular button, check if the point is within the circle's radius from its center
        center_x = bx + self.radius
        center_y = by + self.radius
        return distance((x, y), (center_x, center_y)) < self.radius

# --- Layout Setup ---
buttons = []
layout = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['C', '0', '=', '+']
]

start_x = 500  # Right side
start_y = 150
for i in range(4):
    for j in range(4):
        buttons.append(Button(pos=(start_x + j * 90, start_y + i * 90), text=layout[i][j]))

# --- Utility ---
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# --- States ---
equation = ""
pinch_active = False
pinch_start_time = 0
press_cooldown = 0.5
last_press_time = 0

# --- Webcam ---
cap = cv2.VideoCapture(0)
w_cam = 1280
h_cam = 720
cap.set(3, w_cam)
cap.set(4, h_cam)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    now = time.time()

    # Draw display (right side) - Darker background, light text
    cv2.rectangle(img, (start_x, 50), (start_x + 360, 120), (50, 50, 50), cv2.FILLED) # Dark grey background
    cv2.rectangle(img, (start_x, 50), (start_x + 360, 120), (100, 100, 100), 2) # Lighter border
    cv2.putText(img, equation[-15:], (start_x + 10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3) # White text

    # Draw all buttons with rainbow colors
    color_index = 0 # Reset color index for each frame
    for btn in buttons:
        # Use modulo to cycle through rainbow colors
        btn.draw(img, color=RAINBOW_COLORS[color_index % len(RAINBOW_COLORS)], text_color=(0,0,0)) # Black text for visibility
        color_index += 1

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        lm_list = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

        index_tip = lm_list[8]
        thumb_tip = lm_list[4]
        pinch_dist = distance(index_tip, thumb_tip)

        # Draw pointers - unchanged, but can be customized too
        cv2.circle(img, index_tip, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

        if pinch_dist < 40:
            if not pinch_active:
                pinch_start_time = now
                pinch_active = True
            elif now - pinch_start_time > 0.2:
                if now - last_press_time > press_cooldown:
                    for btn in buttons:
                        # Pass the index finger tip coordinates for pressing
                        if btn.is_pressed(*index_tip): 
                            val = btn.text
                            if val == 'C':
                                equation = ""
                            elif val == '=':
                                try:
                                    equation = str(eval(equation))
                                except:
                                    equation = "Err"
                            else:
                                if equation == "Error": # Clear "Error" on new input
                                    equation = ""
                                equation += val
                            last_press_time = now
                            break
        else:
            pinch_active = False

    cv2.imshow("Gesture Calculator", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()