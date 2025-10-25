from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO(r"E:\pycharm pro\PyCharm 2025.1.2\yolo\fall_dataset_train_3rd\exp_6th2\weights\best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取原始帧
    ret, frame = cap.read()
    if not ret:
        break

    # 直接推理
    results = model(frame)

    # 绘制结果
    annotated = results[0].plot()

    # 显示
    cv2.imshow("YOLO Detection", annotated)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
