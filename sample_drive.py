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
CONTROL_HOST = '0.0.0.0'
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
script_start_time = None # Tracks elapsed system time for speed scaling calculations
darkness_consecutive_ticks = 0 # Prevents transient camera glitches from triggering reverse gear

# ==================================================================
# 📊 TACTICAL STATE MACHINE CHECKLIST
# Tracks completion parameters for the +60 tactical win condition
# ==================================================================
tactical_checklist = {
    'darkness_passed':     False,
    'police_passed':        False,
    'chasing_a_passed':    False,
    'chasing_b_passed':    False,
    'golden_lane_passed':  False
}

# ==================================================================
# CHALLENGE 2: Chasing Car — Global State Machine Array
# ==================================================================
chasing_car_state = {
    'prev_back_frame':          None,  # Previous back frame for differencing
    'appearances_count':        0,     # Total tracking entries encountered
    'is_chasing_active':        False, # Status flag indicating active evasion deployment
    'evasion_ticks_remaining':  0,     # Maneuver lifecycle ticker countdown
    'detection_cooldown':       0,     # Guard window frame offset limit
    'consecutive_detections':   0      # Debounce consistency verify tracking filter
}
CHASING_DEBOUNCE_FRAMES    = 3      
CHASING_EVASION_TICKS_1ST = 35     
CHASING_EVASION_TICKS_2ND = 25     
CHASING_DETECTION_COOLDOWN = 4000  
CHASING_MOTION_AREA_THRESH = 500   
CHASING_COLOR_AREA_THRESH  = 300   
CHASING_ROI_Y_START        = 0.30  
CHASING_PIXEL_RATIO_THRESH = 0.02  

# ==================================================================
# CHALLENGE 3: Police Car — Global Tracking Variables
# ==================================================================
police_car_state = {
    'is_active':              False,  
    'event_ticks_remaining':  0,      
    'red_token_caught':       False,  
    'detection_cooldown':     0,      
    'consecutive_detections': 0,      
}
POLICE_DEBOUNCE_FRAMES      = 3     
POLICE_EVENT_DURATION_TICKS = 2000  
POLICE_DETECTION_COOLDOWN   = 3000  
POLICE_LIGHT_AREA_THRESH    = 15    
POLICE_LIGHT_MAX_GAP        = 60    
POLICE_BODY_AREA_THRESH     = 600   
POLICE_COLLISION_Y_RATIO    = 0.78  
POLICE_ROI_Y_START          = 0.25  
POLICE_TOKEN_CENTER_TOL    = 15    
POLICE_TOKEN_CAUGHT_Y_RATIO = 0.80  

# ==================================================================
# 🌟 NEW EVENT: Golden Lane — Global Tracking Variables
# ==================================================================
golden_lane_state = {
    'is_active': False,
    'target_lane': None,          # Tracks lane parsed from screen (1=Left, 2=Center, 3=Right)
    'event_ticks_remaining': 0,   # 500 ticks countdown loop
    'detection_cooldown': 0       # Prevents multi-triggering within the same window
}
GOLDEN_EVENT_DURATION_TICKS = 500 
GOLDEN_DETECTION_COOLDOWN   = 1500 

# ---------------------------------------------------------
# Real-Time Scheduling Framework
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
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
# Network Connection Setup
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
    except Exception:
        pass

def read_front_camera_task(): read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')
def read_back_camera_task():  read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

def _detect_chasing_car_motion(current_frame, prev_frame):
    if current_frame is None or prev_frame is None or current_frame.shape != prev_frame.shape:
        return False

    diff = cv2.absdiff(current_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    height, width = thresh.shape
    roi_y = int(height * CHASING_ROI_Y_START)
    roi = thresh[roi_y:, :]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > CHASING_MOTION_AREA_THRESH:
            return True

    white_pixels = cv2.countNonZero(roi)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels > 0 and (white_pixels / total_pixels) > CHASING_PIXEL_RATIO_THRESH:
        return True

    return False

def _detect_chasing_car_color(frame):
    if frame is None:
        return False

    height, width = frame.shape[:2]
    roi_y = int(height * CHASING_ROI_Y_START)
    roi = frame[roi_y:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_dark = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 80, 120]))
    lower_red1, upper_red1 = np.array([0, 120, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([165, 120, 100]), np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 80]), np.array([130, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))

    mask_combined = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_dark, mask_red), mask_blue), mask_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > CHASING_COLOR_AREA_THRESH:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            roi_width = roi.shape[1]
            if roi_width * 0.1 < cx < roi_width * 0.9:
                return True
    return False

def detect_chasing_car(current_frame, prev_frame):
    return _detect_chasing_car_motion(current_frame, prev_frame) and _detect_chasing_car_color(current_frame)

def _detect_police_lightbar(frame):
    if frame is None:
        return False, None

    height, width = frame.shape[:2]
    roi_y = int(height * POLICE_ROI_Y_START)
    roi = frame[roi_y:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 140, 100]), np.array([8, 255, 255])
    lower_red2, upper_red2 = np.array([172, 140, 100]), np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, np.array([95, 140, 100]), np.array([130, 255, 255]))

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_blobs = [cv2.boundingRect(c) for c in contours_red if cv2.contourArea(c) > POLICE_LIGHT_AREA_THRESH]
    blue_blobs = [cv2.boundingRect(c) for c in contours_blue if cv2.contourArea(c) > POLICE_LIGHT_AREA_THRESH]

    for (rx, ry, rw, rh) in red_blobs:
        red_cx, red_cy = rx + rw // 2, ry + rh // 2
        for (bx, by, bw, bh) in blue_blobs:
            blue_cx, blue_cy = bx + bw // 2, by + bh // 2
            if abs(red_cy - blue_cy) < 20 and abs(red_cx - blue_cx) < POLICE_LIGHT_MAX_GAP:
                return True, (red_cx + blue_cx) // 2
    return False, None

def _detect_police_body(frame):
    if frame is None:
        return None

    height, width = frame.shape[:2]
    roi_y = int(height * POLICE_ROI_Y_START)
    roi = frame[roi_y:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 20])
    upper_dark = np.array([180, 90, 110])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < POLICE_BODY_AREA_THRESH:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y + roi_y, w, h)

def _find_red_token(frame):
    if frame is None:
        return False, None, None

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([8, 255, 255])
    lower_red2, upper_red2 = np.array([172, 100, 100]), np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

    mask_red[0:int(height * 0.30), :] = 0
    mask_red[int(height * 0.85):, :] = 0

    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return False, None, None

    x, y, w, h = cv2.boundingRect(largest)
    return True, x + w // 2, (x, y, w, h)

def _detect_golden_lane_text(frame):
    if frame is None:
        return False, None

    height, width = frame.shape[:2]
    ui_roi = frame[0:int(height * 0.20), :]
    hsv = cv2.cvtColor(ui_roi, cv2.COLOR_BGR2HSV)

    mask_text = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
    contours, _ = cv2.findContours(mask_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 120]
    
    if not valid_contours:
        return False, None

    largest_text = max(valid_contours, key=cv2.contourArea)
    tx, ty, tw, th = cv2.boundingRect(largest_text)
    text_center_x = tx + tw // 2

    if text_center_x < int(width * 0.40): return True, 1  
    elif text_center_x > int(width * 0.60): return True, 3  
    else: return True, 2  

def processing_task():
    global shared_data, steering_cooldown, has_saved_debug, chasing_car_state, police_car_state, golden_lane_state, tactical_checklist, script_start_time, darkness_consecutive_ticks

    # ── 1. GLOBAL LATCH MONITOR (CRITICAL RESTART FLUSH) ──
    if keyboard.is_pressed('r'):
        # Flush all tracking dictionaries to clean configurations
        chasing_car_state.update({'prev_back_frame': None, 'appearances_count': 0, 'is_chasing_active': False, 'evasion_ticks_remaining': 0, 'detection_cooldown': 0, 'consecutive_detections': 0})
        police_car_state.update({'is_active': False, 'event_ticks_remaining': 0, 'red_token_caught': False, 'detection_cooldown': 0, 'consecutive_detections': 0})
        golden_lane_state.update({'is_active': False, 'target_lane': None, 'event_ticks_remaining': 0, 'detection_cooldown': 0})
        tactical_checklist.update({'darkness_passed': False, 'police_passed': False, 'chasing_a_passed': False, 'chasing_b_passed': False, 'golden_lane_passed': False})
        script_start_time = time.time()
        darkness_consecutive_ticks = 0
        with data_lock:
            shared_data['steering_input'] = 0.0
            shared_data['acceleration_input'] = 1.0
            shared_data['trailing_car_detected'] = False
        steering_cooldown = 0
        print("[RESET] Clean slate re-instantiated across all state structures.")
        return

    # ── 2. DATA EXTRACTION LAYER (FETCH FRAMES ONCE) ──
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    # ── 3. CHALLENGE 1: LOW LIGHT DETECTOR (ABSOLUTE HIGHEST PRIORITY) ──
    # Runs at the top to clear the blindspot during police or chasing events
    if front_frame is not None:
        gray = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray)
        standard_deviation = np.std(gray) # Gathers pixel contrast variance metrics

        # Genuine environmental darkness has low brightness AND low variance (entire screen dark).
        # Yellow token camera corruption injects high-contrast glitch static (low brightness BUT high variance).
        if average_brightness < 45.0:
            if standard_deviation < 18.0: # True darkness ceiling constraint filter
                darkness_consecutive_ticks += 1
                if darkness_consecutive_ticks >= 12: 
                    print(f"[CHALLENGE 1] Sustained Darkness confirmed! Brightness: {average_brightness:.2f}, Variance: {standard_deviation:.2f}. Recovering light...")
                    tactical_checklist['darkness_passed'] = True
                    with data_lock:
                        shared_data['steering_input'] = 0.0
                        shared_data['acceleration_input'] = -1.0 # Sends recovery pulse
                    return
            else:
                # High contrast deviation signals a yellow camera glitch! Reject and bypass
                darkness_consecutive_ticks = 0
        else:
            darkness_consecutive_ticks = 0

    # Throttle holds modifier configurations
    if script_start_time is not None:
        elapsed_run_time = time.time() - script_start_time
        cooldown_multiplier = min(2.0, 1.0 + (elapsed_run_time / 90.0))
    else:
        cooldown_multiplier = 1.0

    target_acceleration = 0.65 

    # ── CHALLENGE 2: Chasing Car State Managment Loop ──
    if chasing_car_state['detection_cooldown'] > 0:
        chasing_car_state['detection_cooldown'] -= 1
    else:
        chasing_car_state['detection_cooldown'] = 0

    if chasing_car_state['is_chasing_active']:
        chasing_car_state['evasion_ticks_remaining'] -= 1
        if chasing_car_state['evasion_ticks_remaining'] <= 0:
            print(f"[CHALLENGE 2] Evasion complete. Resuming tactical driving.")
            chasing_car_state['is_chasing_active'] = False
            chasing_car_state['detection_cooldown'] = CHASING_DETECTION_COOLDOWN 
            tactical_checklist['chasing_a_passed' if chasing_car_state['appearances_count'] == 1 else 'chasing_b_passed'] = True
            with data_lock:
                shared_data['steering_input'] = 0.0
            steering_cooldown = int(20 * cooldown_multiplier)  
            return
        else:
            with data_lock:
                shared_data['steering_input'] = -1.0   
                shared_data['acceleration_input'] = 1.0 
            return

    if chasing_car_state['appearances_count'] < 2 and chasing_car_state['detection_cooldown'] <= 0:
        prev_back = chasing_car_state['prev_back_frame']
        if back_frame is not None:
            car_detected = detect_chasing_car(back_frame, prev_back)
            chasing_car_state['prev_back_frame'] = back_frame.copy()

            if car_detected:
                chasing_car_state['consecutive_detections'] += 1
                if chasing_car_state['consecutive_detections'] >= CHASING_DEBOUNCE_FRAMES:
                    chasing_car_state['appearances_count'] += 1
                    chasing_car_state['is_chasing_active'] = True
                    chasing_car_state['consecutive_detections'] = 0
                    chasing_car_state['evasion_ticks_remaining'] = CHASING_EVASION_TICKS_1ST if chasing_car_state['appearances_count'] == 1 else CHASING_EVASION_TICKS_2ND
                    with data_lock:
                        shared_data['steering_input'] = -1.0
                        shared_data['acceleration_input'] = 1.0 
                        shared_data['trailing_car_detected'] = True
                    return
            else:
                chasing_car_state['consecutive_detections'] = 0

    with data_lock:
        shared_data['trailing_car_detected'] = False

    # ── 🌟 Proportional Golden Lane Sweeper Execution Loop ──
    if golden_lane_state['detection_cooldown'] > 0:
        golden_lane_state['detection_cooldown'] -= 1

    if golden_lane_state['is_active']:
        golden_lane_state['event_ticks_remaining'] -= 1
        
        if front_frame is not None:
            height, width = front_frame.shape[:2]
            frame_center_x = width // 2
            
            # Use HSV to isolate all green tokens
            hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, np.array([40, 80, 100]), np.array([80, 255, 255]))
            
            # Mask out the sky and the immediate hood to focus on the road
            mask_green[0:int(height * 0.30), :] = 0 
            mask_green[int(height * 0.85):, :] = 0
            
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            target_steering = 0.0
            
            if contours_green:
                # Filter for valid tokens and find the largest one (closest to the car)
                valid_greens = [c for c in contours_green if cv2.contourArea(c) > 80]
                if valid_greens:
                    largest_green = max(valid_greens, key=cv2.contourArea)
                    gx, gy, gw, gh = cv2.boundingRect(largest_green)
                    green_center_x = gx + gw // 2
                    
                    # Proportional control to center the car on the green token streak
                    deviation = green_center_x - frame_center_x
                    
                    # Apply a small deadzone (15 pixels) to prevent steering jitter once aligned
                    if abs(deviation) > 15:
                        target_steering = np.clip(deviation / (width * 0.25), -1.0, 1.0)

            with data_lock:
                shared_data['steering_input'] = target_steering
                # Boost speed to capitalize on the guaranteed +10% green token speed bonuses
                shared_data['acceleration_input'] = 0.85 
                
        # Win condition check: 5 seconds (500 ticks at 100Hz) elapsed
        if golden_lane_state['event_ticks_remaining'] <= 0:
            print("[GOLDEN LANE] Event survived. Target lane secured.")
            golden_lane_state['is_active'] = False
            tactical_checklist['golden_lane_passed'] = True
            golden_lane_state['detection_cooldown'] = GOLDEN_DETECTION_COOLDOWN
            steering_cooldown = 10
        return

    # ── CHALLENGE 3: Police Car Active Tracker Layer ──
    if police_car_state['detection_cooldown'] > 0:
        police_car_state['detection_cooldown'] -= 1

    if police_car_state['is_active']:
        police_car_state['event_ticks_remaining'] -= 1

        if front_frame is not None:
            body_box = _detect_police_body(front_frame)
            token_found, token_cx, token_box = _find_red_token(front_frame)
            frame_width, frame_height = front_frame.shape[1], front_frame.shape[0]
            frame_center_x = frame_width // 2

            target_steering = 0.0
            target_acceleration = 0.65 

            collision_imminent = False
            if body_box is not None:
                bx, by, bw, bh = body_box
                body_cx, body_bottom = bx + bw // 2, by + bh
                if body_bottom > frame_height * POLICE_COLLISION_Y_RATIO:
                    collision_imminent = True
                    target_acceleration = 1.0 
                    target_steering = -1.0 if body_cx >= frame_center_x else 1.0

            if not collision_imminent and token_found:
                if abs(token_cx - frame_center_x) > POLICE_TOKEN_CENTER_TOL:
                    target_steering = 1.0 if token_cx > frame_center_x else -1.0
                else:
                    target_steering = 0.0

                tx, ty, tw, th = token_box
                if abs(token_cx - frame_center_x) < 40 and (ty + th) > frame_height * POLICE_TOKEN_CAUGHT_Y_RATIO:
                    police_car_state['red_token_caught'] = True

            with data_lock:
                shared_data['steering_input'] = target_steering
                shared_data['acceleration_input'] = target_acceleration

        if police_car_state['event_ticks_remaining'] <= 0:
            police_car_state['is_active'] = False
            tactical_checklist['police_passed'] = True
            police_car_state['red_token_caught'] = False
            police_car_state['detection_cooldown'] = POLICE_DETECTION_COOLDOWN
            with data_lock:
                shared_data['steering_input'] = 0.0
            steering_cooldown = int(20 * cooldown_multiplier)  
        return

    # Background hooks verification triggers
    if front_frame is not None:
        if golden_lane_state['detection_cooldown'] <= 0:
            text_found, parsed_lane = _detect_golden_lane_text(front_frame)
            if text_found:
                print(f"[GOLDEN LANE] Event initialization caught! Heading to Lane {parsed_lane}...")
                golden_lane_state['is_active'] = True
                golden_lane_state['target_lane'] = parsed_lane
                golden_lane_state['event_ticks_remaining'] = GOLDEN_EVENT_DURATION_TICKS
                return

        if police_car_state['detection_cooldown'] <= 0:
            lightbar_found, _ = _detect_police_lightbar(front_frame)
            if lightbar_found:
                police_car_state['consecutive_detections'] += 1
                if police_car_state['consecutive_detections'] >= POLICE_DEBOUNCE_FRAMES:
                    police_car_state['is_active'] = True
                    police_car_state['event_ticks_remaining'] = POLICE_EVENT_DURATION_TICKS
                    police_car_state['red_token_caught'] = False
                    police_car_state['consecutive_detections'] = 0
                    return

    # ── Passive Driving Layer ──
    if steering_cooldown > 0:
        steering_cooldown -= 1
        if steering_cooldown == 0:
            with data_lock:
                shared_data['steering_input'] = 0.0
        return

    if front_frame is not None:
        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
        frame_center_x = front_frame.shape[1] // 2
        target_steering = 0.0

        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([8, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([172, 100, 100]), np.array([179, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, np.array([50, 80, 100]), np.array([75, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

        height, width = mask_red.shape[:2]
        mask_red[0:int(height * 0.30), :] = 0; mask_green[0:int(height * 0.30), :] = 0; mask_yellow[0:int(height * 0.30), :] = 0
        mask_red[int(height * 0.85):, :] = 0; mask_green[int(height * 0.85):, :] = 0; mask_yellow[int(height * 0.85):, :] = 0

        if not has_saved_debug:
            cv2.imwrite("debug_1_raw_frame.png", front_frame)
            cv2.imwrite("debug_2_red_mask.png", mask_red)
            cv2.imwrite("debug_3_green_mask.png", mask_green)
            cv2.imwrite("debug_4_yellow_mask.png", mask_yellow)
            has_saved_debug = True

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 400:
                x, y, w, h = cv2.boundingRect(largest_red)
                red_center_x = x + w // 2
                if abs(red_center_x - frame_center_x) < 100:
                    target_steering = -1.0 if red_center_x > frame_center_x else 1.0
                    steering_cooldown = int(15 * cooldown_multiplier)
                    with data_lock:
                        shared_data['steering_input'] = target_steering
                        shared_data['acceleration_input'] = target_acceleration
                    return

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
                steering_cooldown = int(8 * cooldown_multiplier)
                with data_lock:
                    shared_data['steering_input'] = target_steering
                    shared_data['acceleration_input'] = target_acceleration
                return

        if contours_green:
            valid_greens = []
            for c in contours_green:
                area = cv2.contourArea(c)
                if area > 100:  
                    gx, gy, gw, gh = cv2.boundingRect(c)
                    valid_greens.append((area, gx, gy, gw, gh, gy + gh))
            
            if valid_greens:
                valid_greens.sort(key=lambda t: t[5], reverse=True)
                area, gx, gy, gw, gh, token_bottom = valid_greens[0]
                green_center_x = gx + gw // 2

                deviation = green_center_x - frame_center_x
                if abs(deviation) > 5:
                    target_steering = np.clip(deviation / (frame_center_x * 0.45), -1.0, 1.0)
                    with data_lock:
                        shared_data['steering_input'] = target_steering
                        shared_data['acceleration_input'] = target_acceleration
                    return

        with data_lock:
            shared_data['steering_input'] = 0.0
            shared_data['acceleration_input'] = target_acceleration

def send_controls_task():
    global control_conn
    if control_conn is None: return
    with data_lock:
        s = shared_data['steering_input']
        a = shared_data['acceleration_input']
    try:
        control_conn.sendall(struct.pack('ff', s, a))
    except Exception:
        control_conn = None

# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    script_start_time = time.time()

    window_back = "Back Camera Feed (Rear View)"
    back_window_created = False

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.01, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    t_front_camera.start(); t_back_camera.start(); t_processing.start(); t_controls.start()

    try:
        while is_running:
            with data_lock:
                back_frame = shared_data['latest_back_frame']
                trailing_detected = shared_data['trailing_car_detected']

            if back_frame is not None:
                if not back_window_created:
                    cv2.namedWindow(window_back, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_back, 640, 480)
                    back_window_created = True

                back_display = back_frame.copy()
                if trailing_detected or chasing_car_state['is_chasing_active']:
                    h, w = back_display.shape[:2]
                    cv2.rectangle(back_display, (0, 0), (w-1, h-1), (0, 0, 255), 4)
                    cv2.putText(back_display, "!! PURSUIT THREAT !!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if police_car_state['is_active']:
                    h, w = back_display.shape[:2]
                    cv2.rectangle(back_display, (0, 0), (w-1, h-1), (0, 140, 255), 4)

                cv2.imshow(window_back, cv2.resize(back_display, (640, 480)))

            if cv2.waitKey(30) & 0xFF == 27: 
                break
    except KeyboardInterrupt:
        is_running = False

    is_running = False
    t_front_camera.join(); t_back_camera.join(); t_processing.join(); t_controls.join()
    if front_camera_sock: front_camera_sock.close()
    if back_camera_sock: back_camera_sock.close()
    if control_conn: control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")