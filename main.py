import cv2
import math
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS


hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = None
running = False


def classify_distance(dist):
    if dist < 30:
        return "Zoom Out"
    elif dist < 60:
        return "Neutral"
    else:
        return "Zoom In"


def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def count_fingers(landmarks, hand_label):
    finger_tips = [4, 8, 12, 16, 20]
    fingers = []

    
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    
    for i in range(1, 5):
        fingers.append(
            1 if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y else 0
        )

    return sum(fingers)


def detect_gesture(finger_count):
    if finger_count == 0:
        return "Fist"
    elif finger_count == 5:
        return "Open Hand"
    else:
        return "Partial"



def get_all_finger_distances(landmarks, w, h):
    tips = {
        "Thumb": mp_hands.HandLandmark.THUMB_TIP,
        "Index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
        "Middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        "Ring": mp_hands.HandLandmark.RING_FINGER_TIP,
        "Pinky": mp_hands.HandLandmark.PINKY_TIP
    }

    points = {}
    for name, idx in tips.items():
        points[name] = (
            int(landmarks[idx].x * w),
            int(landmarks[idx].y * h)
        )

    distances = {
        "Thumb-Index": euclidean(points["Thumb"], points["Index"]),
        "Index-Middle": euclidean(points["Index"], points["Middle"]),
        "Middle-Ring": euclidean(points["Middle"], points["Ring"]),
        "Ring-Pinky": euclidean(points["Ring"], points["Pinky"]),
        "Thumb-Pinky": euclidean(points["Thumb"], points["Pinky"])
    }

    return points, distances


def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()


def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    video_label.config(image="")


def update_frame():
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness):

            mp_draw.draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)

            label = handedness.classification[0].label
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape

            # Finger count + gesture
            finger_count = count_fingers(landmarks, label)
            gesture = detect_gesture(finger_count)

            # Thumb–Index zoom distance
            thumb_tip = (
                int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w),
                int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h)
            )
            index_tip = (
                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            )

            dist = euclidean(thumb_tip, index_tip)
            zoom_gesture = classify_distance(dist)

            cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} Hand: {gesture}, {zoom_gesture} ({int(dist)} px)",
                (10, 40 if label == "Right" else 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # 🔹 ALL FINGER DISTANCES
            points, distances = get_all_finger_distances(landmarks, w, h)

            # Draw finger lines
            cv2.line(frame, points["Thumb"], points["Index"], (255, 0, 0), 2)
            cv2.line(frame, points["Index"], points["Middle"], (255, 0, 0), 2)
            cv2.line(frame, points["Middle"], points["Ring"], (255, 0, 0), 2)
            cv2.line(frame, points["Ring"], points["Pinky"], (255, 0, 0), 2)
            cv2.line(frame, points["Thumb"], points["Pinky"], (0, 0, 255), 2)

            # Display distances
            y = 120 if label == "Right" else 260
            for name, d in distances.items():
                cv2.putText(
                    frame,
                    f"{name}: {int(d)} px",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                y += 25

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)


root = Tk()
root.title("Hand Gesture Recognition")
root.geometry("900x700")

Label(root, text="Hand Gesture Recognition", font=("Arial", 24)).pack(pady=10)

video_label = Label(root)
video_label.pack()

Button(root, text="Start Camera", command=start_camera, width=15).pack(pady=5)
Button(root, text="Stop Camera", command=stop_camera, width=15).pack(pady=5)

root.mainloop()
