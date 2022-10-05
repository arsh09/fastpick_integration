"""
A quick way to log your network results (latency and throughput). 

Muhammad Arshad 
26-Sept-2022
"""
import  shlex, psutil
import subprocess as sp 
from pping import ping 
import re 
import sys
import time
import csv
import os
import signal

def network_throughput( interface_name , count, ping_ms_time = 0.0 ) : 

    cmd = ["sar", "-n", "DEV", str(count), "1"] 
    cmd_sar = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE ) 

    cmd = ["grep", interface_name ]
    cmd_grep = sp.Popen( cmd, stdin = cmd_sar.stdout, stdout = sp.PIPE, stderr = sp.PIPE )

    cmd_out, cmd_err = cmd_grep.communicate()
    cmd_out = cmd_out.decode('utf-8')

    cmd_out = cmd_out.split('\n')

    latency_and_throughput = ""
    for line in cmd_out : 
        if "Average" not in line:
            line = re.sub(" +", ",", line).split(",")
            line = ",".join(line[:-1])
            latency_and_throughput =  "{},{}".format(line,ping_ms_time)
            break

    return latency_and_throughput.split(",")

def netowrk_latency( ip ) : 

    cmd = ["ping", ip, "-c", "1" ]
    cmd_ping = sp.Popen( cmd, stdout = sp.PIPE, stderr = sp.PIPE ) 

    cmd = ["grep", "icmp_seq=1" ]
    cmd_grep = sp.Popen( cmd, stdin = cmd_ping.stdout, stdout = sp.PIPE, stderr = sp.PIPE )

    cmd_out, cmd_err = cmd_grep.communicate() 
    cmd_out = cmd_out.decode("utf-8")

    cmd_out = re.sub(" +", ",", cmd_out).split(",")

    ms_time = 0.0
    for line in cmd_out : 
        if "time=" in line: 
            line = line.split("=") 
            ms_time = line[-1]

    return ms_time

if __name__ == "__main__":

    if len( sys.argv ) != 4: 
        print ("\nUsage: \npython network_test.py <interface-name> <other-computer-ip-addr> <counts-in-seconds>\n\n")
        sys.exit()

    interface_name, ip_addr, counts_in_seconds = sys.argv[1], sys.argv[2], sys.argv[3]

    headers = "time,interface,rxpck/s,txpck/sec,rxkB/s,txkB/s,rxcmp/s,txcmp/s,rxmcst/s,%ifutil,latency"

    network_data = []
    network_data.append( headers.split(",") )

    # start a rosbag 
    cmd = "rosbag record /camera/image_raw/compressed -O ./{}-{}-{}".format(interface_name, ip_addr, counts_in_seconds)
    cmd = shlex.split(cmd)
    # cmd_rosbag = sp.Popen( cmd )

    for count in range( int( counts_in_seconds )  ):
        try:
            ping_ms_time = netowrk_latency( ip_addr )
            value = network_throughput( interface_name , 1, ping_ms_time)
            network_data.append(value)    
            print ( "Count: {} - {}".format(count, value) ) 
            time.sleep(0.5)
        except KeyboardInterrupt: 
            break


    csv_filename = "./network_test_{}_{}_{}.csv".format( interface_name , ip_addr.replace(".", "-"), counts_in_seconds )
    with open(csv_filename, "w") as f: 
        wr = csv.writer(f)
        wr.writerows(network_data)

    print("Network test is saved in {}".format(csv_filename))

    # for proc in psutil.process_iter():
    #     if "record" in proc.name() and set(cmd[2:]).issubset(proc.cmdline()):
    #         proc.send_signal(sp.signal.SIGINT)

    # cmd_rosbag.send_signal(sp.signal.SIGINT)