import cv2
import mediapipe as mp
import urllib.request
import os
import time

# 1. Automatically download the AI model file if it doesn't exist
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the hand landmark model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Set up the modern MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# 3. Find the right camera
print("Searching for a camera...")
cap = None
# Linux sometimes maps webcams to 0, 2, or requires the V4L2 backend
for camera_index in [0, 2, 1, -1]:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Success! Camera found at index {camera_index}.")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("\n❌ ERROR: Could not open any camera.")
    print("Troubleshooting tips for Linux:")
    print("1. Check if another app (like Zoom or a browser) is using the camera.")
    print("2. Run 'ls -l /dev/video*' in your terminal to see connected cameras.")
    print("3. Ensure your user has permissions to access the camera (try adding your user to the 'video' group).")
    exit()

# 4. Initialize the Landmarker and start the webcam
last_timestamp_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    print("Camera is active! Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera read failed. Exiting loop.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Safeguard: MediaPipe crashes if two frames have the exact same timestamp
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        # Detect hands
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw the landmarks
        if result.hand_landmarks:
            h, w, _ = frame.shape

            for hand_landmarks in result.hand_landmarks:
                pixel_landmarks = [(int(lm.x * w), int(lm.y * h))
                                   for lm in hand_landmarks]

                for x, y in pixel_landmarks:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for connection in HAND_CONNECTIONS:
                    start_idx = connection.start
                    end_idx = connection.end
                    
                    if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                        cv2.line(
                            frame, 
                            pixel_landmarks[start_idx], 
                            pixel_landmarks[end_idx], 
                            (255, 0, 0), 
                            2
                        )

        cv2.imshow('Modern Hand Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
