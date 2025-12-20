import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

finger_tips = [4, 8, 12, 16, 20]

while True:
    ret, img = cam.read()
    view = cv2.flip(img, 1)
    result = hands.process(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mpDraw.draw_landmarks(view, hand_landmarks, mpHands.HAND_CONNECTIONS)
            label = handedness.classification[0].label
            h, w, _ = view.shape
            landmarks = hand_landmarks.landmark
            finger_count = 0

            if label == "Right":
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0]-1].x:
                    finger_count += 1
            else:
                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0]-1].x:
                    finger_count += 1

            for tip in finger_tips[1:]:
                if landmarks[tip].y < landmarks[tip-2].y:
                    finger_count += 1

            cv2.putText(view, f"{label} Hand - Fingers: {finger_count}",
                        (10, 50 if label == "Right" else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand detection", view)
    if cv2.waitKey(1) == ord('q'):
        break

hands.close()
cam.release()
cv2.destroyAllWindows()
