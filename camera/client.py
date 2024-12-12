import socket

def send_data():
    host = '192.168.10.10'
    port = 12345

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    while True:
        user_input = input("Enter a message: ")
        s.sendall(user_input.encode())

try:
    send_data()
except KeyboardInterrupt:
    print("Stopped by User")