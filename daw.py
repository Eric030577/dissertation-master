

import socket
client = socket.socket()  # 默认是AF_INET、SOCK_STREAM
client.connect(('192.168.31.238',6868))
s = "ee"
client.send(s.encode("utf-8"))
while True:

    data = str(client.recv(1024))
    print(data)

