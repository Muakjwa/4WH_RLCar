import cv2
import numpy as np
import pickle
from multiprocessing import Process, Queue, Lock
import time
import socket

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
    
    # ?? ???? ? ???
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if (imwrite):
        cv2.imwrite(f'./naive_image/line{idx}.jpg', line_image)
    
    # ??? ???? ??? ?? ??
    height, width = edges.shape
    center_x = width // 2
    for y in range(height - 1, -1, -1):  # ???? ?? ??
        if edges[y, center_x] > 0:
            return height - y  # ?????? ?? ??
    return 240  # ??? ???? ??

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


if __name__ == '__main__':
    host = '192.168.10.10'
    port = 12345

    left = [-1, -1, -1, -1, -1, -1, 8, 23, 34, 46, 61, 72, 83, 91, 100, 109, 116, 122, 129, 135, 141, 147, 151, 156, 160, 164, 168, 171, 174, 178, 182, 185, 187, 189, 191, 193, 195, 197, 209, 201, 203, 204, 205, 207, 209, 210, 211, 212, 215, 215, 216]
    right = [-1, -1, -1, -1, -1, -1, 13, 26, 41, 53, 65, 76, 86, 96, 104, 113, 121, 128, 134, 141, 147, 152, 157, 161, 166, 170, 174, 178, 181, 184, 186, 189, 193, 195, 197, 200, 202, 204, 206, 208, 209, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221]
    front_right = [-1, -1, -1, -1, -1, -1, 2, 16, 28, 16, 28, 40, 53, 62, 70, 77, 83, 89, 95, 101, 106, 110, 114, 117, 121, 124, 127, 129, 132, 134, 137, 138, 140, 141, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159]
    front_left = [-1, -1, -1, -1, -1, -1, 15, 29, 11, 23, 34, 45, 55, 64, 72, 77, 83, 89, 93, 99, 103, 107, 111, 115, 118, 122, 124, 127, 129, 131, 133, 135, 138, 140, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 160, 161, 162, 163, 200]
    measurement = [left, right, front_right, front_left]

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # Queues for communication
    frame_queue = Queue(maxsize=3)
    result_queue = Queue()
    lock = Lock()

    device = ['/dev/video0', '/dev/video4', '/dev/video8', '/dev/video25']

    # Start camera processes
    processes = []
    imwrite = True
    for idx, dev in enumerate(device):
        p = Process(target=process_camera, args=(
            idx, dev, result_queue, lock, imwrite
        ))
        processes.append(p)
        p.start()

    try:
        while True:
            #time.sleep(0.05)
            # Measure time for each iteration
            start_time = time.time()
            results = {}
            radar_value = []

            # Collect results from all cameras
            for _ in range(len(device)):
                try:
                    idx, distance = result_queue.get(timeout=2)  # Wait for each result
                    results[idx] = distance
                except Exception:
                    print("Timeout waiting for camera result.")
                    continue

            # Print results
            #for idx in sorted(results.keys()):
            for idx in [1, 2, 3, 0]:
                if len(results) == 4:
                    print(f"Camera {idx}: Distance = {results[idx]}")
                    if idx == 0:
                        radar_value.append(find_cm(measurement[idx], results[idx])-1 if find_cm(measurement[idx], results[idx]) >= 1 else find_cm(measurement[idx], results[idx]))
                    elif idx == 1:
                        radar_value.append(find_cm(measurement[idx], results[idx])-1 if find_cm(measurement[idx], results[idx]) >= 1 else find_cm(measurement[idx], results[idx]))
                    elif idx == 2:
                        radar_value.append(find_front_right_cm(measurement[idx], results[idx]) - 6 if find_front_right_cm(measurement[idx], results[idx]) >= 6 else find_front_right_cm(measurement[idx], results[idx]))
                    else:
                        radar_value.append(find_front_right_cm(measurement[idx], results[idx]) - 4 - 6 if find_front_right_cm(measurement[idx], results[idx]) >= 10 else find_front_right_cm(measurement[idx], results[idx]))
            
            if (radar_value):
                try:
                    s.sendall(','.join(map(str, radar_value)).encode())
                except KeyboardInterrupt:
                    print("Stopped by User")
                    # Stop processes
                    for p in processes:
                        p.terminate()

    except KeyboardInterrupt:
        print("Stopping processes...")
        for p in processes:
            p.terminate()

    # Stop processes
    for p in processes:
        p.terminate()
