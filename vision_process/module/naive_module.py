import cv2
import numpy as np
import time

def process_camera(idx, device, result_queue, lock, imwrite):
    time.sleep(0.05)
    print(device)
    cap = cv2.VideoCapture(device)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 8)

    if not cap.isOpened():
        print(f"Camera {device} failed to open.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        distance = process_image(img, imwrite, idx)

        # Send the result back to the main process
        with lock:
            result_queue.put((idx, distance))

def custom_grayscale(image):
    # 채널 값이 230 이상이면 유지, 그렇지 않으면 0
    processed_image = np.where(image >= 240, image, 0)
    
    # R, G, B 채널 값을 평균하여 그레이스케일 변환
    grayscale_image = np.mean(processed_image, axis=2).astype(np.uint8)
    return grayscale_image

def process_image(image, imwrite = False, idx = 0):
    if (imwrite):
        cv2.imwrite(f'./naive_image/original{idx}.jpg', image)
    # Gaussian Blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    if (imwrite):
        cv2.imwrite(f'./naive_image/blur1{idx}.jpg', blurred)

    # 변환 적용
    grayscale_image = custom_grayscale(blurred)  
    if (imwrite):
        cv2.imwrite(f'./naive_image/customgray{idx}.jpg', grayscale_image)
   
    # Canny Edge Detection
    edges = cv2.Canny(grayscale_image, 100, 150)
    if (imwrite):
        cv2.imwrite(f'./naive_image/canny{idx}.jpg', edges)
    
    # Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10)
    
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if (imwrite):
        cv2.imwrite(f'./naive_image/line{idx}.jpg', line_image)
    
    height, width = edges.shape
    center_x = width // 2
    for y in range(height - 1, -1, -1):
        if edges[y, center_x] > 0:
            return height - y 
    return 240  

def find_cm(measurement, value):
    for i, dis in enumerate(measurement):
        if i > 20:
            return -1
        if value < dis:
            return i
    return 20

def find_front_right_cm(measurement, value):
    for i, dis in enumerate(measurement):
        if i > 20:
            return 20
        if value < dis:
            return i
    return 20

def find_front_left_cm(measurement, value):
    for i, dis in enumerate(measurement):
        if i > 20:
            return 20
        if value < dis:
            return i
    return 20
