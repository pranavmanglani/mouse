import cv2
import mediapipe as mp
import pyautogui
import math

# --- CONFIGURATION & SAFETY ---
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  

# Get screen resolution
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

left_clicked = False
right_clicked = False
CLICK_THRESHOLD = 45 

# --- JITTER FILTER VARIABLES ---
prev_x, prev_y = 0, 0
SMOOTH_FACTOR = 0.15

def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Open Webcam
cap = cv2.VideoCapture(0)

# Create the window first so we can apply flags to it
WINDOW_NAME = "Pure Gesture Mouse Control"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# This line keeps the window pinned on top of all other applications
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

print("\n--- Hand Gesture Control Activated ---")
print(f"Current Click Threshold: {CLICK_THRESHOLD} pixels")
print("Camera GUI pinned to top. Gliding mouse smoothly...\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = {}
            for id, lm in enumerate(hand_landmarks.landmark):
                landmarks[id] = (int(lm.x * frame_w), int(lm.y * frame_h))

            thumb = landmarks.get(4)
            index = landmarks.get(8)
            middle = landmarks.get(12)

            if thumb and index and middle:
                # 1. CURSOR MOVEMENT (Controlled by the Index Finger Tip)
                padding = 60
                scaled_x = int((index[0] - padding) * screen_w / (frame_w - padding * 2))
                scaled_y = int((index[1] - padding) * screen_h / (frame_h - padding * 2))
                
                scaled_x = max(0, min(screen_w - 1, scaled_x))
                scaled_y = max(0, min(screen_h - 1, scaled_y))
                
                # --- APPLY THE ANTI-JITTER FILTER ---
                if prev_x == 0 and prev_y == 0:
                    curr_x, curr_y = scaled_x, scaled_y
                else:
                    curr_x = prev_x + (scaled_x - prev_x) * SMOOTH_FACTOR
                    curr_y = prev_y + (scaled_y - prev_y) * SMOOTH_FACTOR
                
                prev_x, prev_y = curr_x, curr_y

                # Move mouse using the newly smoothed coordinates
                pyautogui.moveTo(int(curr_x), int(curr_y))
                cv2.circle(frame, index, 10, (255, 0, 0), cv2.FILLED)

                # Calculate current live distances
                left_dist = get_distance(index, thumb)
                right_dist = get_distance(middle, thumb)

                # DEBUG LOGGER
                print(f"LIVE DISTANCE -> Left (Index-Thumb): {int(left_dist)} px | Right (Middle-Thumb): {int(right_dist)} px", end="\r")

                # 2. LEFT CLICK GESTURE (Index Finger + Thumb Pinch)
                if left_dist < CLICK_THRESHOLD:
                    if not left_clicked:
                        pyautogui.click()
                        left_clicked = True
                        print(f"\n[ACTION] Left Click Triggered! (Distance: {int(left_dist)})")
                    cv2.circle(frame, index, 12, (0, 255, 0), cv2.FILLED)
                else:
                    left_clicked = False

                # 3. RIGHT CLICK GESTURE (Middle Finger + Thumb Pinch)
                if right_dist < CLICK_THRESHOLD:
                    if not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True
                        print(f"\n[ACTION] Right Click Triggered! (Distance: {int(right_dist)})")
                    cv2.circle(frame, middle, 12, (255, 0, 255), cv2.FILLED)
                else:
                    right_clicked = False

    # Draw frame into our specific pinned window name
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()