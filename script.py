import cv2
import numpy as np
from PIL import ImageGrab
import threading
import time
import logging
import keyboard
from pynput.mouse import Controller

class PlayerDetector:
    def __init__(self, yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0, average_player_height_cm=180.34):
        try:
            self.net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
            self.layer_names = self.net.getUnconnectedOutLayersNames()
        except cv2.error as e:
            logging.error(f"Error initializing YOLO: {e}")
            raise SystemExit(e)

        self.aimbot_strength = aimbot_strength
        self.average_player_height_cm = average_player_height_cm
        self.lock = threading.Lock()
        self.is_aimbot_enabled = False
        self.auto_fire_enabled = False
        self.boxes = []

    def detect_players(self, frame, center):
        if not self.is_aimbot_enabled:
            return frame

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        try:
            outs = self.net.forward(self.layer_names)
        except cv2.error as e:
            logging.error(f"Error in YOLO forward pass: {e}")
            return frame

        class_ids = []
        confidences = []
        boxes = []

        with self.lock:
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and class_id == 0:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        head_top = int(center_y - h * self.aimbot_strength)
                        head_bottom = int(center_y)
                        head_left = int(center_x - w * self.aimbot_strength / 2)
                        head_right = int(center_x + w * self.aimbot_strength / 2)

                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([head_left, head_top, head_right - head_left, head_bottom - head_top])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if indices is not None:
            self.boxes = [boxes[i] for i in indices.flatten()]

            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                if center is not None:
                    center[0] += x + w // 2 - center[0]
                    center[1] += y + h // 2 - center[1]

                # Draw stickman
                self.draw_stickman(frame, x, y, w, h)

                # Draw a red rectangle around the player within a normal radius
                normal_radius = max(w, h) // 2
                cv2.rectangle(frame, (x - normal_radius, y - normal_radius), (x + w + normal_radius, y + h + normal_radius), (0, 0, 255), 2)

                # Auto-fire if enabled
                if self.auto_fire_enabled:
                    self.auto_fire()

        return frame

    def draw_stickman(self, frame, x, y, w, h):
        # Calculate body parts positions
        head_center = (x + w // 2, y)
        shoulder_center = (x + w // 2, y + h // 4 * 3)
        hip_center = (x + w // 2, y + h)

        # Draw stickman
        cv2.circle(frame, head_center, 5, (0, 255, 0), -1)
        cv2.line(frame, head_center, shoulder_center, (0, 255, 0), 2)
        cv2.line(frame, shoulder_center, hip_center, (0, 255, 0), 2)

    def auto_fire(self):
        # Implement your auto-fire logic here
        # For demonstration purposes, simulate left mouse button click
        mouse = Controller()
        mouse.click(Controller.Button.left)

    def adjust_aimbot_strength(self, center):
        if self.aimbot_strength == 1.0 and center is not None and len(self.boxes) > 0:
            # Lock onto the player head if aimbot strength is full
            player_x, player_y, player_w, player_h = self.boxes[0]
            center[0] += player_x + player_w // 2 - center[0]
            center[1] += player_y + player_h // 2 - center[1]

    def calculate_distance(self, box):
        # Calculate the distance based on the average player height
        _, _, w, h = box
        player_height_in_frame = max(w, h)
        distance_multiplier = self.average_player_height_cm / player_height_in_frame
        return distance_multiplier

    def get_nearest_player_distance(self):
        if len(self.boxes) > 0:
            distances = [self.calculate_distance(box) for box in self.boxes]
            return min(distances)
        else:
            return None

class FortniteTracker:
    def __init__(self, window_coordinates, yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0, average_player_height_cm=180.34):
        self.window_coordinates = window_coordinates
        self.detector = PlayerDetector(yolo_weights_path, yolo_cfg_path, aimbot_strength, average_player_height_cm)
        self.tracking_event = threading.Event()
        self.tracking_thread = threading.Thread(target=self.track_players, daemon=True)
        self.window_center = [sum(coord) // 2 for coord in window_coordinates]

    def start_tracking(self):
        self.tracking_event.set()
        if not self.tracking_thread.is_alive():
            self.tracking_thread.start()

    def stop_tracking(self):
        self.tracking_event.clear()
        self.tracking_thread.join()

    def toggle_aimbot(self):
        with self.detector.lock:
            self.detector.is_aimbot_enabled = not self.detector.is_aimbot_enabled

    def toggle_auto_fire(self):
        with self.detector.lock:
            self.detector.auto_fire_enabled = not self.detector.auto_fire_enabled

    def track_players(self):
        try:
            while self.tracking_event.is_set():
                try:
                    screen = self.capture_screen()
                    if screen is None:
                        continue

                    detected_screen = self.detector.detect_players(screen, self.window_center)

                    with self.detector.lock:
                        self.detector.aimbot_strength = 1.0
                        self.detector.is_aimbot_enabled = True
                        self.detector.detect_players(screen, self.window_center)

                        if self.detector.boxes:
                            self.detector.adjust_aimbot_strength(self.window_center)

                    # Get the nearest player distance and display it in meters
                    nearest_player_distance = self.detector.get_nearest_player_distance()
                    if nearest_player_distance is not None:
                        cv2.putText(detected_screen, f"Nearest Player: {nearest_player_distance:.2f}m", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display aimbot strength in the corner
                    cv2.putText(detected_screen, f"Aimbot Strength: {self.detector.aimbot_strength * 100:.0f}%",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow('Fortnite Tracker', detected_screen)

                except Exception as e:
                    logging.error(f"Error in tracking loop: {e}")
                    time.sleep(1)  # Sleep to prevent excessive error messages

                finally:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            logging.error(f"Error in track_players thread: {e}")

    def capture_screen(self):
        try:
            screen = ImageGrab.grab(bbox=(self.window_coordinates[0][0], self.window_coordinates[0][1],
                                          self.window_coordinates[1][0], self.window_coordinates[1][1]))
            return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.error(f"Error capturing screen: {e}")
            return None

def main():
    print("Instructions: Press F9 to toggle aimbot, press F10 to toggle auto-fire.")
    time.sleep(6)  # Initial delay
    game_window = [(0, 0), (1920, 1080)]  # Update with your game window coordinates
    yolo_weights_path = "path/to/your/custom/model.weights"
    yolo_cfg_path = "path/to/your/custom/model.cfg"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if YOLO model is loaded successfully
    try:
        tracker = FortniteTracker(game_window, yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0)
    except SystemExit as e:
        logging.error(f"Exiting due to initialization error: {e}")
        return

    try:
        keyboard.add_hotkey('F9', tracker.toggle_aimbot)
        keyboard.add_hotkey('F10', tracker.toggle_auto_fire)

        tracker.start_tracking()

        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        tracker.stop_tracking()
        cv2.destroyAllWindows()

def run():
    main()
