import time
from dataclasses import dataclass
from pathlib import Path
import yaml
import cv2
import mediapipe as mp


# Base directory of the current script
BASE_DIR = Path(__file__).resolve().parent

# Default model path for the MediaPipe gesture recognizer
MODEL_PATH = BASE_DIR / 'gesture_recognizer.task'

# Supported gestures and output colors used by the app
GESTURES = ['Open_Palm', 'Closed_Fist', 'Thumb_Up']
COLOR_NAMES = ['red', 'green', 'blue']

# OpenCV uses BGR color ordering, not RGB
BGR_COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
}


# Load the global configuration from cfg.yml
CFG_PATH = BASE_DIR / 'cfg.yml'
with open(CFG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

mediapipe_cfg = cfg["mediapipe"]

@dataclass
class GestureColorMapping:
    # Stores the user-defined gesture assigned to each target color
    red: str
    green: str
    blue: str


def invert_mapping(mapping: GestureColorMapping):
    # Convert color -> gesture mapping into gesture -> color mapping
    # so detected gestures can be translated directly into cube colors
    return {
        mapping.red: 'red',
        mapping.green: 'green',
        mapping.blue: 'blue',
    }


def create_gesture_recognizer(model_path: Path):
    # Build the MediaPipe gesture recognizer with fixed options
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=mediapipe_cfg["num_hands"],
        min_hand_detection_confidence=mediapipe_cfg["min_hand_detection_confidence"],
        min_hand_presence_confidence=mediapipe_cfg["min_hand_presence_confidence"],
        min_tracking_confidence=mediapipe_cfg["min_tracking_confidence"]
    )
    return GestureRecognizer.create_from_options(options)


class GesturePerception:
    def __init__(self, mapping: GestureColorMapping, model_path: Path = MODEL_PATH, stable_frames: int = 8):
        # Check that the model exists before creating the recognizer
        if not model_path.exists():
            raise FileNotFoundError(
                f'Model file not found: {model_path}\n'
                'Download gesture_recognizer.task and place it next to perception.py.'
            )

        # Save the user-selected gesture mapping
        self.mapping = mapping
        self.gesture_to_color = invert_mapping(mapping)

        # Create MediaPipe recognizer
        self.recognizer = create_gesture_recognizer(model_path)

        # Number of consecutive frames required before triggering a command
        self.stable_frames = stable_frames

        # Internal debounce state
        self.last_color = None
        self.stable_count = 0

        # Start time used to generate monotonically increasing timestamps in ms
        self.start_time = time.time()

    def process_frame(self, frame):
        # Mirror the webcam image so interaction feels natural to the user
        frame = cv2.flip(frame, 1)

        # Convert BGR frame from OpenCV to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe VIDEO mode requires a monotonically increasing timestamp in ms
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        detected_gesture = None
        detected_color = None
        triggered_color = None

        # Read the top gesture if at least one gesture prediction is available
        if result.gestures and len(result.gestures[0]) > 0:
            detected_gesture = result.gestures[0][0].category_name

        # Convert the detected gesture into a mapped color
        if detected_gesture in self.gesture_to_color:
            detected_color = self.gesture_to_color[detected_gesture]

            # Debounce: require the same mapped color for several consecutive frames
            if detected_color == self.last_color:
                self.stable_count += 1
            else:
                self.last_color = detected_color
                self.stable_count = 1

            # Trigger the command only when the stability threshold is reached
            if self.stable_count == self.stable_frames:
                triggered_color = detected_color
        else:
            # Reset debounce state when no mapped gesture is detected
            self.last_color = None
            self.stable_count = 0

        # Draw visualization on the frame for user feedback
        self._draw_overlay(frame, detected_gesture, detected_color)
        return frame, detected_gesture, detected_color, triggered_color

    def _draw_overlay(self, frame, detected_gesture, detected_color):
        # Render the current recognition status on top of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        if detected_color is not None:
            color_bgr = BGR_COLORS[detected_color]
            cv2.putText(frame, f'Gesture: {detected_gesture}', (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, detected_color.upper(), (20, 90), font, 1.5, color_bgr, 4, cv2.LINE_AA)
            cv2.putText(frame, f'Stable: {self.stable_count}/{self.stable_frames}', (20, 130), font, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Perform a mapped gesture', (20, 40), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    def close(self):
        # Release MediaPipe resources cleanly
        self.recognizer.close()