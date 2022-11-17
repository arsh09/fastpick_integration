import random
import socket, select
from time import gmtime, strftime
from random import randint

imgcounter = 1
basename = "image.png"

HOST = '127.0.0.1'
PORT = 1243

connected_clients_sockets = []

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(10)

connected_clients_sockets.append(server_socket)
# print('Entering loop: ')

try:
    while True:

        print('while true')
        read_sockets, write_sockets, error_sockets = select.select(connected_clients_sockets, [], [])
        print('in loop')

        for sock in read_sockets:

            print('in sock')

            if sock == server_socket:

                sockfd, client_address = server_socket.accept()
                connected_clients_sockets.append(sockfd)

                print ('if in sock')

            else:
                try:

                    # print('else')

                    data = sock.recv(4096)
                    txt = str(data, "utf-8")
                    # txt = str(data)

                    print('data')

                    if data:

                        print('if data: ', type(data))

                        if data.startswith(b'SIZE'):

                            print('if data starts with: ', txt)
                            tmp = txt.split()
                            size = int(tmp[1])

                            print('size: ', size)
                            #
                            print('got size')

                            sock.sendall(bytes("GOT SIZE", "utf-8"))

                        elif data.startswith(b'BYE'):
                            print('sock shut down')
                            sock.shutdown()

                        else:

                            print('else data else data')

                            myfile = open(basename, 'wb')
                            myfile.write(data)

                            print('openned file')

                            # data = sock.recv(40960000)
                            data = sock.recv(1506)
                            print('data received')
                            if not data:
                                print('no data')
                                myfile.close()
                                break
                            myfile.write(data)
                            print('data written')
                            myfile.close()

                            sock.sendall("GOT IMAGE")
                            sock.shutdown()
                except:
                    sock.close()
                    connected_clients_sockets.remove(sock)
                    continue
            imgcounter += 1
finally:
    server_socket.close()
    print('server socket closed')
