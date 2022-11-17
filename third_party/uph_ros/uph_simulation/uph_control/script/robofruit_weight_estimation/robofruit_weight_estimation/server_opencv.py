# Welcome to PyShine

# This code is for the server
# Lets import the libraries
import socket, cv2, pickle, struct, imutils

# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 9999
socket_address = (host_ip, port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)

data = b""
payload_size = struct.calcsize("Q")

# Socket Accept
# while True:
#     client_socket, addr = server_socket.accept()
#     print('GOT CONNECTION FROM:', addr)
#     if client_socket:
#         vid = cv2.VideoCapture(0)
#
#         while (vid.isOpened()):
#             img, frame = vid.read()
#             frame = imutils.resize(frame, width=320)
#             a = pickle.dumps(frame)
#             message = struct.pack("Q", len(a)) + a
#             client_socket.sendall(message)
#
#             cv2.imshow('TRANSMITTING VIDEO', frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 client_socket.close()
client_socket, addr = server_socket.accept()

while True:

    while len(data) < payload_size:
        # print('first while')
        packet = client_socket.recv(1024 * 4)  # 4K
        # print('packet: ', packet)
        if not packet: break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    print('packed_msg_size', packed_msg_size)
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    print('data len', len(data), msg_size)
    while len(data) < msg_size:
        data += client_socket.recv(64000)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data, encoding="bytes")
    client_socket.send(bytes('frame received', 'utf-8'))

    cv2.imshow('RECEIVING FRAME', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break



