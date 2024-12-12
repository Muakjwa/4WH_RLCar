import pickle
from multiprocessing import Process, Queue, Lock
import time
import socket
from module.main_module import *

if __name__ == '__main__':
    host = '192.168.10.10'
    port = 12345

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    
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

    except KeyboardInterrupt:
        print("Stopping processes...")

    # Stop processes
    for p in processes:
        p.terminate()
