import cv2
import mediapipe as mp
import pyautogui
import time

# -----------------------------
# SETTINGS
# -----------------------------

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MOVE_SENSITIVITY = 1000
SMOOTHING = 0.08

BLINK_THRESHOLD = 0.23
SCROLL_THRESHOLD = 120

DEADZONE = 3

# -----------------------------
# MEDIAPIPE
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# LANDMARK IDS
# -----------------------------

LEFT_EYE = 468
RIGHT_EYE = 473

LEFT_TOP = 159
LEFT_BOTTOM = 145
LEFT_LEFT = 33
LEFT_RIGHT = 133

RIGHT_TOP = 386
RIGHT_BOTTOM = 374
RIGHT_LEFT = 362
RIGHT_RIGHT = 263

NOSE_ID = 1
FOREHEAD_ID = 10

# -----------------------------
# VARIABLES
# -----------------------------

calibrated = False
start_time = time.time()

base_x = 0
base_y = 0

smooth_dx = 0
smooth_dy = 0

left_down = False
right_down = False

scroll_mode = False
scroll_start_y = 0

# -----------------------------
# FUNCTION
# -----------------------------

def eye_ratio(top, bottom, left, right, h, w):

    top_y = int(top.y * h)
    bottom_y = int(bottom.y * h)

    left_x = int(left.x * w)
    right_x = int(right.x * w)

    eye_height = abs(top_y - bottom_y)
    eye_width = abs(left_x - right_x)

    if eye_width == 0:
        return 1

    return eye_height / eye_width

# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # -----------------------------
        # LANDMARKS
        # -----------------------------

        nose = face_landmarks.landmark[NOSE_ID]
        forehead = face_landmarks.landmark[FOREHEAD_ID]

        left_eye = face_landmarks.landmark[LEFT_EYE]
        right_eye = face_landmarks.landmark[RIGHT_EYE]

        # -----------------------------
        # HEAD DIFFERENCE
        # -----------------------------

        current_x = left_eye.z - right_eye.z
        current_y = forehead.z - nose.z

        # -----------------------------
        # DRAW POINTS
        # -----------------------------

        lx = int(left_eye.x * w)
        ly = int(left_eye.y * h)

        rx = int(right_eye.x * w)
        ry = int(right_eye.y * h)

        nx = int(nose.x * w)
        ny = int(nose.y * h)

        fx = int(forehead.x * w)
        fy = int(forehead.y * h)

        cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(frame, (rx, ry), 4, (255, 0, 0), -1)
        cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)
        cv2.circle(frame, (fx, fy), 4, (255, 255, 0), -1)

        # -----------------------------
        # CALIBRATION
        # -----------------------------

        if not calibrated:

            remain = 5 - (time.time() - start_time)

            cv2.putText(
                frame,
                f"Keep Face Still: {remain:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if time.time() - start_time > 5:

                base_x = current_x
                base_y = current_y

                calibrated = True

        else:

            # -----------------------------
            # CURSOR MOVEMENT
            # -----------------------------

            dx = -(current_x - base_x) * MOVE_SENSITIVITY
            dy = -(current_y - base_y) * MOVE_SENSITIVITY

            # -----------------------------
            # DEADZONE
            # -----------------------------

            if abs(dx) < DEADZONE:
                dx = 0

            if abs(dy) < DEADZONE:
                dy = 0

            # -----------------------------
            # SMOOTHING
            # -----------------------------

            smooth_dx = smooth_dx + (dx - smooth_dx) * SMOOTHING
            smooth_dy = smooth_dy + (dy - smooth_dy) * SMOOTHING

            pyautogui.moveRel(
                int(smooth_dx),
                int(smooth_dy)
            )

            # -----------------------------
            # BLINK DETECTION
            # -----------------------------

            left_blink = eye_ratio(
                face_landmarks.landmark[LEFT_TOP],
                face_landmarks.landmark[LEFT_BOTTOM],
                face_landmarks.landmark[LEFT_LEFT],
                face_landmarks.landmark[LEFT_RIGHT],
                h,
                w
            ) < BLINK_THRESHOLD

            right_blink = eye_ratio(
                face_landmarks.landmark[RIGHT_TOP],
                face_landmarks.landmark[RIGHT_BOTTOM],
                face_landmarks.landmark[RIGHT_LEFT],
                face_landmarks.landmark[RIGHT_RIGHT],
                h,
                w
            ) < BLINK_THRESHOLD

            # -----------------------------
            # STATUS TEXT
            # -----------------------------

            cv2.putText(
                frame,
                f"Left Blink: {left_blink}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right Blink: {right_blink}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # -----------------------------
            # SCROLL MODE
            # -----------------------------

            if left_blink and right_blink and not scroll_mode:

                scroll_mode = True
                scroll_start_y = current_y

            elif scroll_mode and not left_blink and not right_blink:

                scroll_amount = (current_y - scroll_start_y) * 10000

                if abs(scroll_amount) > SCROLL_THRESHOLD:
                    pyautogui.scroll(int(-scroll_amount))

                scroll_mode = False

            # -----------------------------
            # LEFT CLICK
            # -----------------------------

            elif left_blink and not right_blink:

                if not left_down:
                    pyautogui.mouseDown(button="left")
                    left_down = True

            else:

                if left_down:
                    pyautogui.mouseUp(button="left")
                    left_down = False

            # -----------------------------
            # RIGHT CLICK
            # -----------------------------

            if right_blink and not left_blink:

                if not right_down:
                    pyautogui.mouseDown(button="right")
                    right_down = True

            else:

                if right_down:
                    pyautogui.mouseUp(button="right")
                    right_down = False

    # -----------------------------
    # SHOW WINDOW
    # -----------------------------

    cv2.imshow("Face Cursor", frame)

    # EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # REDUCE CPU USAGE
    time.sleep(0.01)

# -----------------------------
# CLEANUP
# -----------------------------

cap.release()
cv2.destroyAllWindows()

#  python gesture_scrol.py