import cv2
import numpy as np

frame = cv2.imread("debug_1_raw_frame.png")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Click on a pixel to print its HSV values
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        print(f"Clicked at ({x},{y}) → HSV: H={pixel[0]}, S={pixel[1]}, V={pixel[2]}")

cv2.namedWindow("Click on a token")
cv2.setMouseCallback("Click on a token", on_mouse)
cv2.imshow("Click on a token", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()