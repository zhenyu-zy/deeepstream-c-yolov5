import cv2
import numpy as np

# 标定板的尺寸
board_width = 10  # 棋盘格的列数
board_height = 10  # 棋盘格的行数
square_size = 0.025  # 每个方块的物理尺寸，单位：米

# 准备标定板的世界坐标点 (假设棋盘格是平的)
obj_points = np.zeros((board_height * board_width, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * square_size

# 存储图像中检测到的角点和相应的世界坐标点
obj_points_list = []
img_points_list = []

# 打开相机
cap = cv2.VideoCapture(0)

# 获取相机的分辨率
ret, frame = cap.read()
if not ret:
    print("无法读取相机图像")
    cap.release()
    cv2.destroyAllWindows()
    exit()

height, width = frame.shape[:2]  # 图像的高度和宽度

# 相机的焦距 (假设已经知道或者通过标定获得)
# 如果没有焦距数据，可以假设一个合适的初值
focal_length = 1.0  # 这里假设焦距为1，可以根据需要调整

# 计算视场角（单位：度）
fov_x = 2 * np.arctan(width / (2 * focal_length)) * (180.0 / np.pi)  # 水平视场角
fov_y = 2 * np.arctan(height / (2 * focal_length)) * (180.0 / np.pi)  # 垂直视场角

print(f"Horizontal FOV: {fov_x} degrees")
print(f"Vertical FOV: {fov_y} degrees")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

    if ret:
        # 如果找到了角点，绘制它们
        cv2.drawChessboardCorners(frame, (board_width, board_height), corners, ret)

        # 保存对象点和图像点
        obj_points_list.append(obj_points)
        img_points_list.append(corners)

    # 显示标定图像
    cv2.imshow("Chessboard", frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 标定相机
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points_list, img_points_list, gray.shape[::-1], None, None)

# 打印相机内参和畸变系数
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

cap.release()
cv2.destroyAllWindows()
