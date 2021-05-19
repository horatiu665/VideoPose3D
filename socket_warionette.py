import socket
import numpy as np

class WarionetteSocket:

    def __init__(self, port=26004):
        self.port = port
        self.serverAddressPort = ("127.0.0.1", self.port)
        self.bufferSize = 1024

        # connect right away
        self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        print("Connected to localhost port " + str(port))

    def send_positions(self, pos_array):
        # convert to bytes?
        # then send?
        what = 100

    def send_bytes(self, bytes_to_send: bytes):
        self.UDPClientSocket.sendto(bytes_to_send, self.serverAddressPort)

    def close(self):
        self.UDPClientSocket.close()
