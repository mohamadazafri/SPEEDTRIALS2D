import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame':    None,
    'latest_back_frame':     None,
    'steering_input':        0.0,
    'acceleration_input':    0.0,
    'trailing_car_detected': False
}
data_lock = threading.Lock()
is_running = True

# Global Tracking Variables for the Logic
steering_cooldown = 0   # Counts down how long a "tap" should last
has_saved_debug = False # Flag to save debug images only once

# ==================================================================
# CHALLENGE 2: Chasing Car — Global Tracking Variables
# ==================================================================
# The chasing car appears TWICE during the game.
# 1st appearance: player has 10 seconds to avoid it.
# 2nd appearance: player has only 3 seconds to avoid it.
# If it collides with the player → 50% speed loss.
#
# Strategy: detect motion in the back camera via frame differencing.
#           When detected, immediately steer left to evade.
# ==================================================================
chasing_car_state = {
    'prev_back_frame': None,         # Previous back frame for differencing
    'appearances_count': 0,          # How many times the chasing car has appeared (max 2)
    'is_chasing_active': False,      # True while a chasing car event is ongoing
    'evasion_ticks_remaining': 0,    # Countdown ticks for the evasion manoeuvre
    'detection_cooldown': 0,         # Cooldown after an evasion to avoid re-triggering
    'consecutive_detections': 0,     # Consecutive frames with motion detected (debounce)
}
# Timing constants (1 tick = 5ms task period)
CHASING_DEBOUNCE_FRAMES   = 3      # Only 3 consecutive detections needed (~15ms) — react FAST
CHASING_EVASION_TICKS_1ST = 500    # 500 ticks × 5ms = 2.5s evasion steering for 1st car
CHASING_EVASION_TICKS_2ND = 300    # 300 ticks × 5ms = 1.5s evasion steering for 2nd car
CHASING_DETECTION_COOLDOWN = 4000  # 4000 ticks × 5ms = 20s cooldown between appearances
CHASING_MOTION_AREA_THRESH = 500   # Min motion contour area (px²) — lowered for sensitivity
CHASING_COLOR_AREA_THRESH  = 300   # Min color contour area (px²) for car-color detection
CHASING_ROI_Y_START        = 0.30  # Examine bottom 70% of back frame (wider scan)
CHASING_PIXEL_RATIO_THRESH = 0.02  # If >2% of ROI pixels show motion, car is present

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')


# ==================================================================
# CHALLENGE 2: Chasing Car — Detection Functions
# ==================================================================

def _detect_chasing_car_motion(current_frame, prev_frame):
    """METHOD 1: Frame differencing — detect motion in the back camera.
    Returns True if significant motion is found in the ROI."""
    if current_frame is None or prev_frame is None:
        return False
    if current_frame.shape != prev_frame.shape:
        return False
    
    diff = cv2.absdiff(current_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)  # Lower threshold (was 25)
    
    # Full-width ROI, bottom 70% of frame
    height, width = thresh.shape
    roi_y = int(height * CHASING_ROI_Y_START)
    roi = thresh[roi_y:, :]
    
    # Method 1a: Contour-based check
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > CHASING_MOTION_AREA_THRESH:
            return True
    
    # Method 1b: Pixel ratio check — even without large contours,
    # if many pixels are changing, something big is approaching
    white_pixels = cv2.countNonZero(roi)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels > 0 and (white_pixels / total_pixels) > CHASING_PIXEL_RATIO_THRESH:
        return True
    
    return False


def _detect_chasing_car_color(frame):
    """METHOD 2: Color-based detection — look for car-colored objects.
    
    The chasing car is typically a distinct colored vehicle. We look for
    dark/metallic objects AND bright-colored objects that are NOT part of
    the normal road/sky background.
    
    Returns True if a car-like colored object is found in the ROI."""
    if frame is None:
        return False
    
    height, width = frame.shape[:2]
    roi_y = int(height * CHASING_ROI_Y_START)
    roi = frame[roi_y:, :]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Look for dark/metallic objects (car body — low saturation, low-mid value)
    # This catches dark gray, black, dark blue cars
    lower_dark = np.array([0, 0, 30])
    upper_dark = np.array([180, 80, 120])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Look for bright saturated objects (red, blue, white cars)
    # Red car (wraps around 0°)
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # Blue car
    lower_blue = np.array([100, 100, 80])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # White/bright car
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine all car-color masks
    mask_combined = cv2.bitwise_or(mask_dark, mask_red)
    mask_combined = cv2.bitwise_or(mask_combined, mask_blue)
    mask_combined = cv2.bitwise_or(mask_combined, mask_white)
    
    # Morphological operations to clean up noise and merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > CHASING_COLOR_AREA_THRESH:
            # Additional check: the contour should be roughly in the centre
            # (a car chasing from behind approaches from the middle)
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            roi_width = roi.shape[1]
            # Check if contour centre is within the middle 80% of frame
            if roi_width * 0.1 < cx < roi_width * 0.9:
                return True
    
    return False


def detect_chasing_car(current_frame, prev_frame):
    """Combined chasing car detection using multiple methods.
    Returns True if ANY method detects a car behind us."""
    motion_detected = _detect_chasing_car_motion(current_frame, prev_frame)
    color_detected = _detect_chasing_car_color(current_frame)
    
    # Require BOTH methods to agree to avoid false positives:
    # - Motion alone triggers on road scrolling (false positive)
    # - Color alone triggers on static scenery (false positive)
    # - Both together = a new colored object that's also moving = real car
    return motion_detected and color_detected

def processing_task():
    global shared_data, steering_cooldown, has_saved_debug, chasing_car_state
    
    # ==================================================================
    # CHALLENGE 2: Chasing Car — Back Camera Processing (HIGHEST PRIORITY)
    # ==================================================================
    # Check the back camera for an approaching chasing car.
    # This runs before everything else because collision = 50% speed loss.
    # ==================================================================
    
    # Tick down detection cooldown regardless of other state
    if chasing_car_state['detection_cooldown'] > 0:
        chasing_car_state['detection_cooldown'] -= 1
    
    # If currently evading, count down and keep steering
    if chasing_car_state['is_chasing_active']:
        chasing_car_state['evasion_ticks_remaining'] -= 1
        
        if chasing_car_state['evasion_ticks_remaining'] <= 0:
            # Evasion manoeuvre complete — return to centre
            print(f"[CHALLENGE 2] Evasion complete for appearance #{chasing_car_state['appearances_count']}. Resuming normal driving.")
            chasing_car_state['is_chasing_active'] = False
            chasing_car_state['detection_cooldown'] = CHASING_DETECTION_COOLDOWN
            with data_lock:
                shared_data['steering_input'] = 0.0
            steering_cooldown = 20  # Brief pause before normal processing resumes
            return
        else:
            # Still evading — hold the evasion steering
            with data_lock:
                shared_data['steering_input'] = -1.0   # Steer hard left to dodge
                shared_data['acceleration_input'] = 1.0 # Maintain speed during evasion
            return
    
    # Only attempt detection if we haven't used up both appearances
    # and the cooldown has expired
    if (chasing_car_state['appearances_count'] < 2 
            and chasing_car_state['detection_cooldown'] <= 0):
        
        with data_lock:
            back_frame = shared_data['latest_back_frame']
        
        prev_back = chasing_car_state['prev_back_frame']
        
        if back_frame is not None:
            car_detected = detect_chasing_car(back_frame, prev_back)
            chasing_car_state['prev_back_frame'] = back_frame.copy()
            
            if car_detected:
                chasing_car_state['consecutive_detections'] += 1
                
                # Debounce: require CHASING_DEBOUNCE_FRAMES consecutive detections
                if chasing_car_state['consecutive_detections'] >= CHASING_DEBOUNCE_FRAMES:
                    chasing_car_state['appearances_count'] += 1
                    chasing_car_state['is_chasing_active'] = True
                    chasing_car_state['consecutive_detections'] = 0
                    
                    # Set evasion duration based on which appearance this is
                    if chasing_car_state['appearances_count'] == 1:
                        chasing_car_state['evasion_ticks_remaining'] = CHASING_EVASION_TICKS_1ST
                        print(f"[CHALLENGE 2]  Chasing car DETECTED (1st appearance)! "
                              f"Evading for {CHASING_EVASION_TICKS_1ST * 5}ms. "
                              f"Player has 10s window.")
                    else:
                        chasing_car_state['evasion_ticks_remaining'] = CHASING_EVASION_TICKS_2ND
                        print(f"[CHALLENGE 2]  Chasing car DETECTED (2nd appearance)! "
                              f"Evading for {CHASING_EVASION_TICKS_2ND * 5}ms. "
                              f"Player has only 3s window!")
                    
                    # Immediately start evasion
                    with data_lock:
                        shared_data['steering_input'] = -1.0
                        shared_data['acceleration_input'] = 1.0
                        shared_data['trailing_car_detected'] = True
                    return
            else:
                # Reset consecutive detection counter
                chasing_car_state['consecutive_detections'] = 0
    
    # Reset trailing car flag when not actively chasing
    with data_lock:
        shared_data['trailing_car_detected'] = False
    
    # ==================================================================
    # Normal processing continues below (steering cooldown, Challenge 1, etc.)
    # ==================================================================
    
    if steering_cooldown > 0:
        steering_cooldown -= 1
        if steering_cooldown == 0:
            with data_lock:
                shared_data['steering_input'] = 0.0
        return

    with data_lock:
        front_frame = shared_data['latest_front_frame']
    
    if front_frame is not None:
        # ==================================================================
        # CHALLENGE 1: LOW LIGHT BRIGHTNESS DETECTION
        # ==================================================================
        # Convert to grayscale to isolate pure pixel value brightness intensity
        gray = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray)
        
        # If the average pixel value falls below the dark threshold, signal the headlights!
        # Normal game view is bright blue city/road; Low Light drops near black.
        if average_brightness < 45.0: 
            print(f"[CHALLENGE 1] Low Light detected! Brightness: {average_brightness:.2f}. Recovering light...")
            with data_lock:
                shared_data['steering_input'] = 0.0
                shared_data['acceleration_input'] = -1.0 # Reverse command to reset headlights
            return
        # ==================================================================

        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
        frame_center_x = front_frame.shape[1] // 2
        
        target_steering = 0.0
        target_acceleration = 1.0
        
        lower_red1 = np.array([0,   100, 180])
        upper_red1 = np.array([8,   255, 255])
        lower_red2 = np.array([172, 100, 180])
        upper_red2 = np.array([179, 255, 255])
        lower_green = np.array([50, 80, 180])
        upper_green = np.array([75, 255, 255])
        lower_yellow = np.array([20, 100, 180])
        upper_yellow = np.array([35, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        height, width = mask_red.shape[:2]
        mask_red[0:int(height * 0.30), :] = 0
        mask_green[0:int(height * 0.30), :] = 0
        mask_yellow[0:int(height * 0.30), :] = 0
        
        mask_red[int(height * 0.85):, :] = 0
        mask_green[int(height * 0.85):, :] = 0
        mask_yellow[int(height * 0.85):, :] = 0
        
        if not has_saved_debug:
            cv2.imwrite("debug_1_raw_frame.png", front_frame)
            cv2.imwrite("debug_2_red_mask.png", mask_red)
            cv2.imwrite("debug_3_green_mask.png", mask_green)
            cv2.imwrite("debug_4_yellow_mask.png", mask_yellow)
            print("[DEBUG] Saved snapshots to your project folder!")
            has_saved_debug = True
        
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # PRIORITY 1: Dodge Red Tokens
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 400:
                x, y, w, h = cv2.boundingRect(largest_red)
                red_center_x = x + w // 2
                
                if abs(red_center_x - frame_center_x) < 100:
                    if red_center_x > frame_center_x:
                        target_steering = -1.0
                    else:
                        target_steering = 1.0
                    
                    steering_cooldown = 15
                    
                    with data_lock:
                        shared_data['steering_input'] = target_steering
                        shared_data['acceleration_input'] = target_acceleration
                    return

        # PRIORITY 2: Dodge Yellow Tokens
        danger_yellows = []
        for c in contours_yellow:
            area = cv2.contourArea(c)
            if area > 200:
                x, y, w, h = cv2.boundingRect(c)
                danger_yellows.append((area, x, y, w, h, y + h))

        if danger_yellows:
            danger_yellows.sort(key=lambda t: t[5], reverse=True)
            area, x, y, w, h, token_bottom = danger_yellows[0]
            yellow_center_x = x + w // 2

            if abs(yellow_center_x - frame_center_x) < 130:
                target_steering = -1.0 if yellow_center_x >= frame_center_x else 1.0
                steering_cooldown = 8
                print(f"[YELLOW] Dodging yellow token at x={yellow_center_x}, steering={target_steering}")
                with data_lock:
                    shared_data['steering_input'] = target_steering
                    shared_data['acceleration_input'] = target_acceleration
                return

        # PRIORITY 3: Chase Green Tokens
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 300:
                x, y, w, h = cv2.boundingRect(largest_green)
                green_center_x = x + w // 2
                
                if green_center_x > frame_center_x + 30:
                    target_steering = 1.0
                    steering_cooldown = 10
                elif green_center_x < frame_center_x - 30:
                    target_steering = -1.0
                    steering_cooldown = 10
                    
                if target_steering != 0.0:
                    with data_lock:
                        shared_data['steering_input'] = target_steering
                        shared_data['acceleration_input'] = target_acceleration
                    return

        with data_lock:
            shared_data['steering_input'] = 0.0
            shared_data['acceleration_input'] = target_acceleration

def send_controls_task():
    global control_conn
    if control_conn is None:
        return
    
    with data_lock:
        steering_input = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        control_conn = None

# ---------------------------------------------------------
# Main (Scheduler Initialization & Safe Main Window Loop)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # ==================================================================
        # SAFE RENDERING ZONE: Main thread handles OpenCV GUI window outputs
        # ==================================================================
        while is_running:
            with data_lock:
                back_frame = shared_data['latest_back_frame']
                trailing_detected = shared_data['trailing_car_detected']
            
            # Displays the back camera perspective safely without causing thread freezes
            if back_frame is not None:
                back_display = back_frame.copy()
                
                # ==================================================================
                # CHALLENGE 2: Visual indicator on back camera when chasing car detected
                # ==================================================================
                if trailing_detected or chasing_car_state['is_chasing_active']:
                    # Draw red warning border
                    h, w = back_display.shape[:2]
                    cv2.rectangle(back_display, (0, 0), (w-1, h-1), (0, 0, 255), 4)
                    cv2.putText(back_display, "!! CHASING CAR DETECTED !!",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    app_num = chasing_car_state['appearances_count']
                    ticks_left = chasing_car_state['evasion_ticks_remaining']
                    cv2.putText(back_display, f"Appearance #{app_num} | Evading: {ticks_left*5}ms left",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                back_resized = cv2.resize(back_display, (400, 300))
                cv2.imshow("Back Camera Feed (Monitoring Trail Link)", back_resized)
            
            # Use short waitKey delay to process rendering updates smoothly
            if cv2.waitKey(30) & 0xFF == 27: # Esc key breaks loop cleanly
                break
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    is_running = False
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")