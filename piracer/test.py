import serial
import threading
from action_list import actions

teensy_ports = ['/dev/LeftBack', '/dev/LeftFront', '/dev/RightBack', '/dev/RightFront']
teensies = [serial.Serial(port, 9600, timeout=1) for port in teensy_ports]


def send_command(teensy, command):
    teensy.write(f"{command}\n".encode())


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

def read_serial_input():
    while True:
        action = input("Enter action (STRAIGHT, MOVE_FORWARD, LEFT_TURN, RIGHT_TURN): ").strip().upper()
        perform_action(action)

try:
    print("Waiting for serial input...")
    read_serial_input()
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    for teensy in teensies:
        teensy.close()
