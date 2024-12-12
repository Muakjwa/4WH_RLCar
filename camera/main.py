import cv2
import numpy as np
import pickle
from multiprocessing import Process, Queue, Lock
import time
import socket

def undistort_image(image, calibration_data):
    K = calibration_data['K']
    D = calibration_data['D']
    h, w = image.shape[:2]

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img

def generate_lut(world_x_min, world_x_max, world_x_interval,
                 world_y_min, world_y_max, world_y_interval,
                 K, extrinsic_matrix):
    world_x_coords = np.arange(world_x_min, world_x_max, world_x_interval)
    world_y_coords = np.arange(world_y_min, world_y_max, world_y_interval)

    output_height = len(world_y_coords)
    output_width = len(world_x_coords)

    xv, yv = np.meshgrid(world_x_coords, world_y_coords)
    ones = np.ones_like(xv)

    world_points = np.stack((xv, yv, np.zeros_like(xv), ones), axis=-1).reshape(-1, 4).T  # shape: (4, N)

    # ??? ???? ??
    camera_points = extrinsic_matrix @ world_points  # shape: (3, N)

    # ??? ???? ??
    image_points = K @ camera_points  # shape: (3, N)
    image_points /= image_points[2, :]  # ???

    u_coords = image_points[0, :].reshape((output_height, output_width))
    v_coords = image_points[1, :].reshape((output_height, output_width))

    map_x = u_coords.astype(np.float32)
    map_y = v_coords.astype(np.float32)

    return map_x, map_y

def crop_polygon(image, points, angle=0):
    hull = cv2.convexHull(np.array(points, dtype=np.int32))

    # ??? ??
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # ???? ???? ??? ??
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    # Bounding Rect? ???? ??? ??
    x, y, w, h = cv2.boundingRect(hull)
    cropped_image = cropped_image[y:y+h, x:x+w]

    # ??? ??
    if angle != 0:
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated_image
    else:
        return cropped_image

def process_camera(idx, device, calibration_data, roi, point, lut, frame_queue, result_queue, lock, imwrite = False):

    time.sleep(0.05)
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

        # Undistort the image
        undistort_img = undistort_image(img, calibration_data)

        # Bird?s Eye View transformation
        map_x, map_y = lut
        bev_image = cv2.remap(undistort_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (imwrite):
            cv2.imwrite(f'./image/bev_image{idx}.jpg', bev_image)

        # Crop ROI and calculate distance
        roi_image = crop_polygon(bev_image, roi)
        if (imwrite):
            cv2.imwrite(f'./image/roi_image{idx}.jpg',roi_image)
        distance = calculate_pixel_distance(roi_image, roi,  point,idx, imwrite = imwrite)

        # Send the result back to the main process
        with lock:
            result_queue.put((idx, distance))

def crop_polygon(image, points):
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(hull)
    return cropped_image[y:y+h, x:x+w]

def calculate_pixel_distance(image,roi_points, base_point, idx,imwrite = False):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX = 0, sigmaY = 0)

    edges = cv2.Canny(blur, 100, 200)
    
    if (imwrite):
        cv2.imwrite(f'./image/edge{idx}.jpg', edges)

    if idx == 0:
        input_line = (7,305,348,376)
    elif idx == 1:
        input_line = (15,280,327,312)
    elif idx == 2:
        input_line = (7,178,661,167)
    else:
        input_line = (2,140,585,146)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_polygon = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_polygon], 255)

    edges_in_roi = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi /180 , threshold=70,minLineLength=10, maxLineGap=10)  

    if lines is None:
        print("No lines detected.")
        return 10000
    
    intersections = []
    x1, y1, x2, y2 = input_line
    for line in lines:
        x3, y3, x4, y4 = line[0]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:  # ??? ??
            continue

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        if cv2.pointPolygonTest(roi_polygon, (px, py), False) >= 0:
            intersections.append((px, py))

    if not intersections:
        # print("No intersections found within ROI.")
        return 2000

    min_distance = float("inf")
    bx, by = base_point
    for px, py in intersections:
        distance = np.hypot(px - bx, py - by)
        if distance < min_distance:
            min_distance = distance

    return min_distance 

if __name__ == '__main__':
    host = '192.168.10.10'
    port = 12345

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    # Camera configurations
    '''
    camera_configs = [
        {'index': 0, 'device': '/dev/video0',
         'world_x_min': -0.0, 'world_x_max': 0.3,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 1, 'device': '/dev/video4',
         'world_x_min': -0.10, 'world_x_max': 0.2,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 2, 'device': '/dev/video8',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.35, 'world_y_max': 0.35,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 3, 'device': '/dev/video25',
         'world_x_min': -0.15, 'world_x_max': 0.25,
         'world_y_min': -0.15, 'world_y_max': 0.3,
         'world_x_interval': 0.001, 'world_y_interval': 0.001}
    ]

    roi_config = [
        [(522, 774), (522, 469), (172, 246), (181, 929)],
        [(67, 137), (44, 496), (426, 4), (322, 593)],
        [(109, 575), (119, 875), (774, 865), (771, 544)],
        [(201, 259), (793, 249), (203, 507), (793, 531)]
    ]
    '''
    
    #roi_config = list(np.array(roi_config) // 2)
    #print(roi_config)

    #distance_point = [(296, 306), (25, 247), (15, 15), (15, 15)]
    #distance_point = list(np.array(distance_point) // 2)
    
    camera_configs = [
        {'index': 0, 'device': '/dev/video0',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.3, 'world_y_max': 0.3,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
    
        {'index': 1, 'device': '/dev/video4',
         'world_x_min': -0.10, 'world_x_max': 0.2,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
        
        {'index': 2, 'device': '/dev/video8',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.35, 'world_y_max': 0.35,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
        
        {'index': 3, 'device': '/dev/video25',
         'world_x_min': -0.15, 'world_x_max': 0.25,
         'world_y_min':-0.15, 'world_y_max': 0.3,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005}
    ]
    
    roi_config = [
        [(522, 774), (522, 469),  (172, 246), (181, 929)],
        [(67, 137), (44, 496), (426, 4), (322, 593)],
        [(109, 575), (119, 875), (774, 865), (771, 544)],
        [(201, 259), (793, 249), (203, 507), (793, 531)]
        ]
    

    distance_point = [(285,378), (30,204), (26,368), (1,249)]

    # Load calibration data
    calibration_data = {}
    lut = {}
    for config in camera_configs:
        idx = config['index']
        with open(f'./params/calibration_data_camera{idx}_2.pkl', 'rb') as f:
            calibration_data[idx] = pickle.load(f)

        # Generate LUT
        calib_data = calibration_data[idx]
        lut[idx] = generate_lut(
            config['world_x_min'], config['world_x_max'], config['world_x_interval'],
            config['world_y_min'], config['world_y_max'], config['world_y_interval'],
            calib_data['K'], calib_data['extrinsic_matrix']
        )

    # Queues for communication
    frame_queue = Queue(maxsize=3)
    result_queue = Queue()
    lock = Lock()

    # Start camera processes
    processes = []
    imwrite = True
    for config in camera_configs:
        idx = config['index']
        device = config['device']
        roi = roi_config[idx]
        point = distance_point[idx]
        p = Process(target=process_camera, args=(
            idx, device, calibration_data[idx], roi, point, lut[idx],
            frame_queue, result_queue, lock, imwrite
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
            for _ in range(len(camera_configs)):
                try:
                    idx, distance = result_queue.get(timeout=2)  # Wait for each result
                    results[idx] = distance
                except Exception:
                    print("Timeout waiting for camera result.")
                    continue

            # Print results
            for idx in sorted(results.keys()):
                if len(results) == 4:
                    print(f"Camera {idx}: Distance = {results[idx]}")
                    radar_value.append(results[idx])
            
            if (radar_value):
                try:
                    s.sendall(','.join(map(str, radar_value)).encode())
                except KeyboardInterrupt:
                    print("Stopped by User")

            # Measure and print elapsed time
           # elapsed_time = time.time() - start_time
            #print(f"Iteration time: {elapsed_time:.4f} seconds")

    except KeyboardInterrupt:
        print("Stopping processes...")

    # Stop processes
    for p in processes:
        p.terminate()
