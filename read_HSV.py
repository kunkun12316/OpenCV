import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param['hsv']
        print(f"HSV value at ({x}, {y}): {hsv[y, x]}")

def main():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Frame")
    hsv_value = {'hsv': None}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value['hsv'] = hsv

        cv2.imshow("Frame", frame)
        cv2.setMouseCallback("Frame", mouse_callback, hsv_value)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
