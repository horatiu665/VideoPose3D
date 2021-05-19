import socket

import numpy as np

localIP = "127.0.0.1"

localPort = 26004

bufferSize = 1024

msgFromServer = "Hello UDP Client"

bytesToSend = str.encode(msgFromServer)

# Create a datagram socket

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip

UDPServerSocket.bind((localIP, localPort))

print("UDP server up and listening on " + str(localPort))

# Listen for incoming datagrams

while (True):
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)

    message = bytesAddressPair[0]

    address = bytesAddressPair[1]

    try:
        message_decoded = message.decode()
    except:
        print("aha! exception.")
        arr = np.frombuffer(message, dtype=np.uint8)
        message_decoded = str(arr)
        print(message_decoded)


    print("Message from Client:{}".format(message))
    print("Formatted:{}".format(message_decoded))

    print("Client IP Address:{}".format(address))

    # Sending a reply to client

    UDPServerSocket.sendto(bytesToSend, address)
