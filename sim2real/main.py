import torch
import serial
import threading
from collections import deque
import socket
import time
from action_list import actions, action_decision, set_posture, set_distance
from model.ppo import *

teensy_ports = ['/dev/LeftBack', '/dev/LeftFront', '/dev/RightBack', '/dev/RightFront']
teensies = [serial.Serial(port, 9600, timeout=1) for port in teensy_ports]

def send_command(teensy, command):
    teensy.write(f"{command}\n".encode())
    while True:
        response = teensy.readline().decode().strip() 
        if response == "READY":
            print(f"{teensy.port}: READY received")
            break

def perform_action(action):
    threads = []

    if action not in actions:
        print(f"Action '{action}' is not defined!")
        return

    commands = actions[action]
    for i, teensy in enumerate(teensies):
        command = commands[i] 
        thread = threading.Thread(target=send_command, args=(teensy, command))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join() 

    print(f"Action '{action}' performed.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_action(model, state):
    perform_action('S')
    ex_pos = ''
    while (True): 
        if not ((len(state[0]) and len(state[1]) and len(state[2]) and len(state[3]))):
            continue
        else:
            state_comb = [list(state[0])[-1], list(state[1])[-1], list(state[2])[-1], list(state[3])[-1]]
        if ex_pos == 'S':
            state_comb.append(0)
        elif ex_pos == 'DR':
            state_comb.append(1)
        elif ex_pos == 'DL':
            state_comb.append(2)
        else: 
            state_comb.append(0)
            
        action = select_action_evaluation(state_comb, model, device)
        
        mode, move = action_decision(action)

        print(f"State : {state_comb} => Action: {mode} {move}")

        pos = set_posture(mode)

        if ex_pos == pos:
            pass
        elif pos =='TR':
            perform_action('DR')
            ex_pos = 'DR'
        elif pos =='TL':
            perform_action('DL')
            ex_pos = 'DL'
        elif pos != '':
            perform_action(pos)
            time.sleep(0.1)
            ex_pos = pos

        dis = set_distance(move)
        if dis != '' and (pos!='TR' and pos!='TL'):
            perform_action(dis)

def start_server(radar_data):
    host = "0.0.0.0"
    port = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    
    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data: 
                break
            print(data.decode())
            got_list = []
            for i, x in enumerate(data.decode().split(',')):
                if x != '-1':
                    value = int(int(x) // 2)
                    radar_data[i].append(value)

if __name__ == "__main__":
    ppo_model_path = './model/ppo_model.pth'
    
    state_size = 5
    action_size = 8  

    model = load_model(ppo_model_path, state_size, action_size, device)

    radar_data = deque([(0,0,0,0),(0,0,0,0),(0,0,0,0)], maxlen = 3)

    radar1 = deque([], maxlen = 3)
    radar2 = deque([], maxlen = 3)
    radar3 = deque([], maxlen = 3)
    radar4 = deque([], maxlen = 3)
    radar_data = [radar1, radar2, radar3, radar4]

    server_thread = threading.Thread(target = start_server, args = (radar_data,))
    piracer_thread = threading.Thread(target = do_action, args = (model, radar_data))

    server_thread.start()
    piracer_thread.start()
    server_thread.join()
    piracer_thread.join()
