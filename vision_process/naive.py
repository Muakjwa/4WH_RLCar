from multiprocessing import Process, Queue, Lock
import time
import socket
from module.naive_module import *

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
