import cv2
import numpy as np


def get_color_mask(hsv, lower_bound, upper_bound):
    return cv2.inRange(hsv, lower_bound, upper_bound)


def detect_objects(frame, hsv, color_ranges):
    objects = []
    for color_name, (lower, upper, obj_num) in color_ranges.items():
        mask = get_color_mask(hsv, lower, upper)

        # 处理 findContours 函数的不同返回值
        contours_info = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:  # 过滤掉小面积的噪声
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, w, h, obj_num))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(obj_num), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return objects


def main():
    cap = cv2.VideoCapture(0)
    color_ranges = {
        "red": (np.array([156, 43, 46]), np.array([180, 255, 255]), 1),  # 红色范围
        "green": (np.array([36, 100, 100]), np.array([86, 255, 255]), 2),  # 绿色范围
        "blue": (np.array([100, 150, 50]), np.array([126, 255, 255]), 3)  # 蓝色范围
    }

    detected_order = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = detect_objects(frame, hsv, color_ranges)

        # 按x坐标排序并显示检测到的物块
        objects.sort(key=lambda obj: obj[0])
        detected_order = [obj[4] for obj in objects]

        # 显示识别顺序
        cv2.putText(frame, f"Detected Order: {detected_order}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
