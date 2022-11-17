#!/usr/bin/python2.7

import random
import socket, select
import numpy as np
from time import gmtime, strftime
from random import randint

image = "/data/localdrive/strawberry_multi_cam/rgb_image/strawberry_20cm_aldi_osaMorocco_classII_C01_S060_V01_W22.4.png"

HOST = '127.0.0.1'
PORT = 1243

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

try:

    # open image
    # myfile = open(image, 'rb')
    data = np.random.randn(1024,720,3)
    data = bytes(data)
    # bytes = myfile.read()
    # size = len(bytes)

    # send image size to server
    sock.sendall(bytes("SIZE " + str(len(data))))
    answer = sock.recv(4096)

    print 'answer = %s' % answer

    # send image to server
    if answer == 'GOT SIZE':

        print('sendall')
        sock.sendall(data)
        print('sent')

        # check what server send
        answer = sock.recv(4096)
        print('receive')
        print 'answer = %s' % answer

        if answer == 'GOT IMAGE' :
            sock.sendall("BYE BYE ")
            print 'Image successfully send to server'

    # myfile.close()
except Exception, exception:
    print str(exception)

finally:
    sock.close()
