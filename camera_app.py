import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def open_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera index {i}")
            return cap
        cap.release()
    print("❌ No working camera found.")
    exit()

model_path = "gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

flash_start = 0
show_image_start = 0
current_image = None
show_flash = False
show_image = False
last_trigger_time = 0
cooldown = 2.5 
gesture_start_time = 0
observing_gesture = None
observe_duration = 0.5  

open_palm_img = cv2.imread("open_palm.jpg")
think_img = cv2.imread("point.jpeg")
fist_img = cv2.imread("fist.jpeg")
point_img = cv2.imread("up.jpg")

def fit_to_frame(frame, image):
    return cv2.resize(image, (frame.shape[1], frame.shape[0]))

def trigger_overlay(image):
    global flash_start, show_flash, current_image, show_image, show_image_start, last_trigger_time
    flash_start = time.time()
    current_image = image
    show_flash = True
    show_image = False
    last_trigger_time = time.time()
    print("⚡ Flash + overlay triggered!")

def handle_result(result):
    global observing_gesture, gesture_start_time, last_trigger_time

    if not result.gestures:
        observing_gesture = None
        return

    gesture = result.gestures[0][0].category_name.lower()

    print(f"🎯 Current gesture: {gesture}", end="\r")

    if time.time() - last_trigger_time < cooldown:
        return

    if observing_gesture != gesture:
        observing_gesture = gesture
        gesture_start_time = time.time()
        return

    if time.time() - gesture_start_time >= observe_duration:
        observing_gesture = None  # reset
        if gesture in ["open_palm", "hands_open", "palm"]:
            trigger_overlay(open_palm_img)
        elif gesture in ["none", "noneb_up"]:
            trigger_overlay(think_img)
        elif gesture in ["pointing_up", "nonething_up"]:
            trigger_overlay(point_img)
        elif gesture in ["closed_fist", "thumb_up"]:
            trigger_overlay(fist_img)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)
recognizer = GestureRecognizer.create_from_options(options)

cap = open_camera()
cap.set(3, 640)
cap.set(4, 480)
print("✅ Gesture Camera started. Hold a gesture for 2 seconds to trigger. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = recognizer.recognize(mp_image)
    handle_result(result)

    if show_flash:
        elapsed = time.time() - flash_start
        if elapsed < 0.3:
            alpha = 1 - (elapsed / 0.3)
            white_overlay = np.full_like(frame, 255, dtype=np.uint8)
            frame = cv2.addWeighted(white_overlay, alpha, frame, 1 - alpha, 0)
        else:
            show_flash = False
            show_image = True
            show_image_start = time.time()

    if show_image and current_image is not None:
        elapsed = time.time() - show_image_start
        if elapsed < 2:
            frame = fit_to_frame(frame, current_image)
        else:
            show_image = False
            current_image = None


    cv2.namedWindow("Gesture Camera", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Gesture Camera", 640, 480)
    cv2.imshow("Gesture Camera", frame)

    cv2.imshow("Gesture Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
