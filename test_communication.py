import socket
import threading
import struct
import cv2
import numpy as np
import keyboard
import time

# Camera (REMOTE HOSTS now)
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082

# Control Server (unchanged)
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

is_running = True


def handle_camera_stream(sock, window_name):
    """
    Receives and displays frames from a connected camera socket.
    """
    print(f"Connected to {window_name}")
    try:
        while True:
            length_bytes = sock.recv(4)
            if not length_bytes:
                print(f"{window_name} disconnected.")
                break

            image_length = int.from_bytes(length_bytes, 'little')

            received_bytes = b''
            while len(received_bytes) < image_length:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    print(f"{window_name} disconnected prematurely.")
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                np_arr = np.frombuffer(received_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    cv2.imshow(window_name, frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except Exception as e:
        print(f"Error in {window_name}: {e}")
    finally:
        sock.close()
        cv2.destroyWindow(window_name)
        print(f"{window_name} closed.")


def connect_to_camera(port, window_name):
    """
    Connects to a remote camera server.
    Retries until connection is successful.
    """
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CAMERA_HOST, port))
            handle_camera_stream(sock, window_name)
            break  # exit if stream ends cleanly
        except ConnectionRefusedError:
            print(f"{window_name} not ready on port {port}, retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"{window_name} connection error: {e}")
            time.sleep(1)


def handle_control_client(conn):
    print(f"Control client connected from {conn.getpeername()}")
    print("Use WASD to control the car. Press 'q' to quit.")

    global is_running
    try:
        while is_running:
            steering_input = 0.0
            acceleration_input = 0.0

            if keyboard.is_pressed('w'):
                acceleration_input = 1.0
            elif keyboard.is_pressed('s'):
                acceleration_input = -1.0

            if keyboard.is_pressed('a'):
                steering_input = -1.0
            elif keyboard.is_pressed('d'):
                steering_input = 1.0

            data = struct.pack('ff', steering_input, acceleration_input)
            conn.sendall(data)

            if keyboard.is_pressed('q'):
                is_running = False
                break

            time.sleep(0.05)

    except Exception as e:
        print(f"Control error: {e}")
    finally:
        conn.close()
        print("Control handler finished.")


def start_control_server():
    global is_running
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((CONTROL_HOST, CONTROL_PORT))
        s.listen()
        print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
        conn, addr = s.accept()
        handle_control_client(conn)
        is_running = False


if __name__ == '__main__':
    # Camera CLIENT threads (connect instead of listen)
    front_camera_thread = threading.Thread(
        target=connect_to_camera,
        args=(FRONT_CAMERA_PORT, "Front Camera")
    )

    back_camera_thread = threading.Thread(
        target=connect_to_camera,
        args=(BACK_CAMERA_PORT, "Back Camera")
    )

    # Control server thread
    control_thread = threading.Thread(target=start_control_server)

    # Start all
    front_camera_thread.start()
    back_camera_thread.start()
    control_thread.start()

    # Wait
    front_camera_thread.join()
    back_camera_thread.join()
    control_thread.join()

    print("All threads finished.")