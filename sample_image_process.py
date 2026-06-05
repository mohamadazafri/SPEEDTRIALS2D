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
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0
}
data_lock = threading.Lock()
is_running = True

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
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key, show_display=True):
    #This function reads the latest frame from the camera socket and stores it in the shared data
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
                
                if show_display:
                    # You may disable this if you don't need to display the frames / This could effect the fps
                    frame_resized = cv2.resize(frame, (640, 480))
                    cv2.imshow(window_name, frame_resized)
                    cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame', show_display=False)

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame', show_display=True)

def processing_task():
    # Retrieve the latest front frame
    with data_lock:
        front_frame = shared_data['latest_front_frame']
    
    if front_frame is not None:
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color (adjust as necessary for the specific shade of green)
        lower_green = np.array([50, 50, 150])
        upper_green = np.array([75, 255, 255])
        
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Ignore the top 25% of the screen to avoid detecting the score text UI
        height, width = mask.shape
        mask[0:int(height * 0.25), :] = 0
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        steering_target = 0.0
        
        if contours:
            # Filter out small noise
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            if valid_contours:
                # Find the maximum area
                max_area = max(cv2.contourArea(c) for c in valid_contours)
                
                # Get contours that have the same (or very similar) size
                same_size_contours = [c for c in valid_contours if max_area - cv2.contourArea(c) <= max_area * 0.05]
                
                # If there are multiple of the same size, choose the one closest to the bottom of the screen
                if len(same_size_contours) > 1:
                    chosen_contour = max(same_size_contours, key=lambda c: cv2.boundingRect(c)[1])
                else:
                    chosen_contour = same_size_contours[0]
                    
                M = cv2.moments(chosen_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    
                    height, width, _ = front_frame.shape
                    center_x = width // 2
                    
                    # Simple Left/Right/Center steering logic without error calculation
                    if cX > center_x + 30:    # Token is on the right
                        steering_target = 1.0
                    elif cX < center_x - 30:  # Token is on the left
                        steering_target = -1.0
                    else:                     # Token is in the center
                        steering_target = 0.0
                    
                    # Clip the steering to valid [-1.0, 1.0] range
                    steering_target = max(-1.0, min(1.0, steering_target))

                    # Draw bounding box around the detected token
                    x, y, w, h = cv2.boundingRect(chosen_contour)
                    cv2.rectangle(front_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw steering arrow at the center of the frame
                    center_x = width // 2
                    center_y = height // 2
                    
                    # Arrow length proportional to steering_target
                    arrow_length = int(steering_target * (width / 2))
                    end_x = center_x + arrow_length
                    end_y = center_y
                    
                    # Only draw arrow if there is a steering target (non-zero length)
                    if arrow_length != 0:
                        cv2.arrowedLine(front_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 5, tipLength=0.3)
                    
                    # Draw a center reference point
                    cv2.circle(front_frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Update the shared data with the calculated steering and acceleration
        with data_lock:
            shared_data['steering_input'] = steering_target
            shared_data['acceleration_input'] = 0.0  # accelerate towards the token

        # Display the processed frame
        frame_resized = cv2.resize(front_frame, (640, 480))
        cv2.imshow("Front Camera", frame_resized)
        cv2.waitKey(1)

def send_controls_task():
    #This is where you send the control commands to the car using the control_conn
    global control_conn
    if control_conn is None:
        return
    
    # Retrieve control values
    with data_lock:
        steering_input = shared_data.get('steering_input', 0.0)
        acceleration_input = shared_data.get('acceleration_input', 1.0)

    try:
        # Pack and send the control command
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
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
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
