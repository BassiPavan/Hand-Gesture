import cv2
import mediapipe as mp
import math

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Gesture thresholds (tune as needed)
def classify_gesture(dist):
    if dist < 30:
        return "Zoom Out"
    elif dist < 60:
        return "Neutral"
    else:
        return "Zoom In"

# Distance between two points
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Webcam loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            h, w, _ = frame.shape
            thumb_tip = int(lm[mp_hands.HandLandmark.THUMB_TIP].x * w), int(lm[mp_hands.HandLandmark.THUMB_TIP].y * h)
            index_tip = int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w), int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            dist = euclidean(thumb_tip, index_tip)
            gesture = classify_gesture(dist)

            # Draw and display
            cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 2)
            cv2.putText(frame, f"{gesture} ({int(dist)} px)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
