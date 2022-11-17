#!/usr/bin/env python
PKG = 'numpy_tutorial'
import roslib;
# roslib.load_manifest(PKG)

import socket, pickle, struct
import sys

import numpy as np
import time

# create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '10.101.12.55'  # paste your server ip address here
port = 9999
# data = b""
payload_size = struct.calcsize("Q")


import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy


def talker(data=b""):
    pub = rospy.Publisher('floats', numpy_msg(Floats))
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        while len(data) < payload_size:
            # print('first while')
            packet = client_socket.recv(1024 * 4)  # 4K
            # print('packet: ', packet)
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(1024)
        bbox_data = data[:msg_size]
        data = data[msg_size:]
        bbox_data = pickle.loads(bbox_data)

        a = numpy.array(bbox_data, dtype=numpy.float32)
        pub.publish(a)
        r.sleep()


if __name__ == '__main__':
    while True:
        # image = cv2.imread(file_name)
        # # image = np.asarray(image, dtype=np.float32)
        # print(type(image))
        # a = pickle.dumps(image)
        # message = struct.pack("Q", len(a)) + a
        # client_socket.send(message)
        # # print(client_socket.recv(1024))
        print('Connecting')
        time.sleep(5)
        client_socket.connect((host_ip, port))  # a tuple

        try:

            while True:
                talker()

        except KeyboardInterrupt as e:
            print('Quitting')
            print('Exception: ', e)
            sys.exit()

        except struct.error as e:
            print('Exception: ', e)
            client_socket.close()

        except ConnectionRefusedError as e:
            print('Exception: ', e)

        except Exception as e:
            print('Exception: ', e)

        finally:
            print('Trying to connect again')
