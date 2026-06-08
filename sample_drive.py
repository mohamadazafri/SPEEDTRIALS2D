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

def processing_task():
    global shared_data, steering_cooldown, has_saved_debug
    
    if steering_cooldown > 0:
        steering_cooldown -= 1
        if steering_cooldown == 0:
            with data_lock:
                shared_data['steering_input'] = 0.0
        return

    with data_lock:
        front_frame = shared_data['latest_front_frame']
    
    if front_frame is not None:
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
        
        # Yellow token HSV threshold parameters (Hue 20-35 range)
        lower_yellow = np.array([20, 100, 180])
        upper_yellow = np.array([35, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) # Yellow color mask
        
        if not has_saved_debug:
            cv2.imwrite("debug_1_raw_frame.png", front_frame)
            cv2.imwrite("debug_2_red_mask.png", mask_red)
            cv2.imwrite("debug_3_green_mask.png", mask_green)
            cv2.imwrite("debug_4_yellow_mask.png", mask_yellow) # Diagnostic yellow channel save
            print("[DEBUG] Saved snapshots to your project folder!")
            has_saved_debug = True
        
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Yellow contours extraction
        
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

        # PRIORITY 2 - Dodge Yellow Tokens (Avoids 5-second dynamic corruption states)
        danger_yellows = []
        for c in contours_yellow:
            area = cv2.contourArea(c)
            if area > 200: # Filter out minor background image artifacts
                x, y, w, h = cv2.boundingRect(c)
                danger_yellows.append((area, x, y, w, h, y + h)) # Track bottom coordinates for proximity sorting

        if danger_yellows:
            # Sort by token_bottom descending = closest yellow token to the car is handled first
            danger_yellows.sort(key=lambda t: t[5], reverse=True)
            area, x, y, w, h, token_bottom = danger_yellows[0]
            yellow_center_x = x + w // 2

            if abs(yellow_center_x - frame_center_x) < 130: # 130px center track safety band
                target_steering = -1.0 if yellow_center_x >= frame_center_x else 1.0
                steering_cooldown = 8 # Fast stabilization recovery step length
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
# Main (Scheduler Initialization)
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
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
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