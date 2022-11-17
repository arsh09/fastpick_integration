# lets make the client code
import socket, cv2, pickle, struct
import numpy as np

# create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '127.0.1.1'  # paste your server ip address here
port = 9999
client_socket.connect((host_ip, port))  # a tuple
data = b""
payload_size = struct.calcsize("Q")

# while True:
#     while len(data) < payload_size:
#         packet = client_socket.recv(4 * 1024)  # 4K
#         if not packet: break
#         data += packet
#     packed_msg_size = data[:payload_size]
#     data = data[payload_size:]
#     msg_size = struct.unpack("Q", packed_msg_size)[0]
#
#     while len(data) < msg_size:
#         data += client_socket.recv(4 * 1024)
#     frame_data = data[:msg_size]
#     data = data[msg_size:]
#     frame = pickle.loads(frame_data)
#     cv2.imshow("RECEIVING VIDEO", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# client_socket.close()

file_name = "/data/localdrive/strawberry_multi_cam/rgb_image/strawberry_20cm_aldi_osaMorocco_classII_C01_S060_V01_W22.4.png"

while True:
    image = cv2.imread(file_name)
    # image = np.asarray(image, dtype=np.float32)
    print(type(image))
    a = pickle.dumps(image)
    message = struct.pack("Q", len(a)) + a
    client_socket.send(message)
    # print(client_socket.recv(1024))

    # Receive packet
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
    print(bbox_data)
# client_socket.close()




