import cv2
import numpy as np

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义HSV中的颜色范围
    lower_color = np.array([35, 100, 100])  # 下限（例如绿色）
    upper_color = np.array([85, 255, 255])  # 上限（例如绿色）

    # 创建颜色掩码
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 将掩码应用于图像
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 显示结果
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
