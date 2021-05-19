import socket
import argparse
import numpy as np
import random

# parse args
parser = argparse.ArgumentParser("Send a message to the server")
parser.add_argument('-m', '--message', default="")
parser.add_argument('-p', '--port', type=int, default=26004)

args = parser.parse_args()


if args.message != "":
    msgFromClient = args.message
    bytesToSend = str.encode(msgFromClient)
else:
    myNumbersList = np.ndarray(shape=(17,3))
    for x in range( myNumbersList.shape[0]):
        for y in range(myNumbersList.shape[1]):
            myNumbersList[x,y] = random.randint(0, 100)
    bytesToSend = myNumbersList.tobytes()
    print("Sending bytes: " + str(bytesToSend))

serverAddressPort = ("127.0.0.1", args.port)

bufferSize = 1024

# Create a UDP socket at client side

print("Connecting to " + str(args.port))

UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Send to server using created UDP socket

print("Sending")

UDPClientSocket.sendto(bytesToSend, serverAddressPort)

print("Waiting for response")

exit()

# response from server
msgFromServer = UDPClientSocket.recvfrom(bufferSize)

msg = "Message from Server {}".format(msgFromServer[0])

print(msg)