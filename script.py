try:
    import cv2
    import numpy as np
    from PIL import ImageGrab
    import threading
    import time
    import logging
    import keyboard
    from pynput.mouse import Controller
    import pygetwindow as gw
    import os

    # Add necessary imports for controller input
    import pygame

    # Initialize pygame for controller input
    pygame.init()
    pygame.joystick.init()

except ImportError as e:
    # If there is an import error, prepend 'lib.' before module and library names
    import lib.cv2 as cv2
    import lib.numpy as np
    from lib.PIL import ImageGrab
    import lib.threading as threading
    import lib.time as time
    import lib.logging as logging
    import lib.keyboard as keyboard
    from lib.pynput.mouse import Controller
    import lib.pygetwindow as gw
    import lib.os as os

    # Add necessary imports for controller input
    import lib.pygame as pygame

class PlayerDetector:
    def __init__(self, yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0, average_player_height_cm=180.34):
        yolo_weights_path = os.path.join("assets", "yolo", yolo_weights_path)
        yolo_cfg_path = os.path.join("assets", "yolo", yolo_cfg_path)

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
                is_player_behind_object = self.is_player_behind_object(frame, x, y, w, h)
                color = (0, 255, 0) if not is_player_behind_object else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                if center is not None:
                    center[0] += x + w // 2 - center[0]
                    center[1] += y + h // 2 - center[1]

                # Draw stickman
                self.draw_stickman(frame, x, y, w, h, color)

                # Draw a rectangle around the player within a normal radius
                normal_radius = max(w, h) // 2
                cv2.rectangle(frame, (x - normal_radius, y - normal_radius),
                              (x + w + normal_radius, y + h + normal_radius), color, 2)

                # Auto-fire if enabled
                if self.auto_fire_enabled:
                    self.auto_fire()

        return frame

    def draw_stickman(self, frame, x, y, w, h, color):
        # Calculate body parts positions
        head_center = (x + w // 2, y)
        shoulder_center = (x + w // 2, y + h // 4 * 3)
        hip_center = (x + w // 2, y + h)

        # Draw stickman
        cv2.circle(frame, head_center, 5, color, -1)
        cv2.line(frame, head_center, shoulder_center, color, 2)
        cv2.line(frame, shoulder_center, hip_center, color, 2)

    def is_player_behind_object(self, frame, x, y, w, h):
        # Example logic: Check if there is a pixel of a certain color in front of the player
        # You might need to customize this based on your actual game environment
        roi = frame[y:y + h, x:x + w]
        color_threshold = 100  # Example threshold, customize based on your environment
        return np.any(roi[:, :, 2] > color_threshold)

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
    def __init__(self, yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0, average_player_height_cm=180.34):
        self.detector = PlayerDetector(yolo_weights_path, yolo_cfg_path, aimbot_strength, average_player_height_cm)
        self.tracking_event = threading.Event()
        self.tracking_thread = threading.Thread(target=self.track_players, daemon=True)
        self.window_center = None
        self.is_tracking_paused = False

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

    def adjust_aimbot_strength_up(self):
        with self.detector.lock:
            self.detector.aimbot_strength = min(1.0, self.detector.aimbot_strength + 0.1)

    def adjust_aimbot_strength_down(self):
        with self.detector.lock:
            self.detector.aimbot_strength = max(0.1, self.detector.aimbot_strength - 0.1)

    def toggle_tracking_pause(self):
        self.is_tracking_paused = not self.is_tracking_paused

    def track_players(self):
        try:
            while self.tracking_event.is_set():
                try:
                    if not self.is_tracking_paused:
                        screen = self.capture_screen()
                        if screen is None:
                            continue

                        detected_screen = self.detector.detect_players(screen, self.window_center)

                        with self.detector.lock:
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

                        # Draw ESP boxes around all detected players
                        self.draw_esp(detected_screen)

                        cv2.imshow('Fortnite Tracker', detected_screen)

                except Exception as e:
                    logging.error(f"Error in tracking loop: {e}")
                    time.sleep(1)  # Sleep to prevent excessive error messages

                finally:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            logging.error(f"Error in track_players thread: {e}")

    def draw_esp(self, frame):
        with self.detector.lock:
            for box in self.detector.boxes:
                x, y, w, h = box
                is_player_behind_object = self.detector.is_player_behind_object(frame, x, y, w, h)
                color = (0, 255, 0) if not is_player_behind_object else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def capture_screen(self):
        try:
            window = gw.getActiveWindow()
            if window is not None:
                self.window_center = [window.left + window.width // 2, window.top + window.height // 2]
                screen = ImageGrab.grab(bbox=(window.left, window.top, window.left + window.width, window.top + window.height))
                return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            else:
                logging.warning("No active window found.")
                return None
        except Exception as e:
            logging.error(f"Error capturing screen: {e}")
            return None

def main():
    print("Instructions: Press F9 to toggle aimbot, press F10 to toggle auto-fire.")
    print("Use UP arrow key to increase aimbot strength and DOWN arrow key to decrease aimbot strength.")
    print("Press F6 to pause/resume tracking.")
    time.sleep(6)  # Initial delay
    yolo_weights_path = "yolov3.weights"
    yolo_cfg_path = "yolov3.cfg"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        tracker = FortniteTracker(yolo_weights_path, yolo_cfg_path, aimbot_strength=1.0)
    except SystemExit as e:
        logging.error(f"Exiting due to initialization error: {e}")
        return

    try:
        keyboard.add_hotkey('F9', tracker.toggle_aimbot)
        keyboard.add_hotkey('F10', tracker.toggle_auto_fire)
        keyboard.add_hotkey('UP', tracker.adjust_aimbot_strength_up)
        keyboard.add_hotkey('DOWN', tracker.adjust_aimbot_strength_down)
        keyboard.add_hotkey('F6', tracker.toggle_tracking_pause)

        # Check if a controller is connected
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            # Check if R1 and R2 buttons are available
            if joystick.get_numbuttons() >= 2:
                print("Controller detected. You can use R1 and R2 buttons together for auto-fire.")
                while True:
                    pygame.event.pump()
                    if joystick.get_button(5) and joystick.get_button(4):
                        tracker.toggle_auto_fire()
                    time.sleep(0.01)
            else:
                print("Controller detected, but not enough buttons for auto-fire.")

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

run()
