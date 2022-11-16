import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.size'] = 14.0
mpl.rcParams['font.family'] = 'Times New Roman'

nrows = 90

#df_mqtt_5g = pd.read_csv("./5MinutesTests/5G-mqtt.csv")
df_mqtt_wifi = pd.read_csv("./wifi_tests/mqtt-wifi-server.csv" , nrows= nrows)


#df_ros_5g = pd.read_csv("./5MinutesTests/5G-mqtt.csv")
df_ros_wifi = pd.read_csv("./wifi_tests/ros-wifi-server.csv", nrows= nrows)



fig = plt.figure(1, figsize=(12,8))
#plt.plot( df_mqtt_5g['latency'] , 'r--' , label = "5G - MQTT")
plt.plot( df_mqtt_wifi['latency'] , 'b--' , label = "WiFi - MQTT")
#plt.plot( df_ros_5g['latency'] , 'g--' , label = "5G - ROS")
plt.plot( df_ros_wifi['latency'] , 'y--' , label = "WiFi - ROS")
plt.legend()
plt.xlabel("Seconds")
plt.ylabel("Latency (in msec.)")
plt.suptitle("Network Latency (Server)")
plt.savefig('./latency_results.pdf')


fig = plt.figure(2, figsize=(12,8))

plt.subplot(2,2,1)
#plt.plot( df_mqtt_5g['rxkB/s'] , 'r--' , label = "5G - MQTT")
plt.plot( df_mqtt_wifi['rxkB/s'] , 'b--' , label = "WiFi - MQTT")
#plt.plot( df_ros_5g['rxkB/s'] , 'g--' , label = "5G - ROS")
plt.plot( df_ros_wifi['rxkB/s'] , 'y--' , label = "WiFi - ROS")
plt.legend()
plt.xlabel("Seconds")
plt.ylabel("RX (in kB/s.)")

plt.subplot(2,2,2)
#plt.plot( df_mqtt_5g['txkB/s'] , 'r--' , label = "5G - MQTT")
plt.plot( df_mqtt_wifi['txkB/s'] , 'b--' , label = "WiFi - MQTT")
#plt.plot( df_ros_5g['txkB/s'] , 'g--' , label = "5G - ROS")
plt.plot( df_ros_wifi['txkB/s'] , 'y--' , label = "WiFi - ROS")
plt.legend()    
plt.xlabel("Seconds")
plt.ylabel("TX (in kB/s.)")

plt.subplot(2,2,3)
#plt.plot( df_mqtt_5g['rxpck/s'] , 'r--' , label = "5G - MQTT")
plt.plot( df_mqtt_wifi['rxpck/s'] , 'b--' , label = "WiFi - MQTT")
#plt.plot( df_ros_5g['rxpck/s'] , 'g--' , label = "5G - ROS")
plt.plot( df_ros_wifi['rxpck/s'] , 'y--' , label = "WiFi - ROS")
plt.legend()
plt.xlabel("Seconds")
plt.ylabel("rxpck/s")


plt.subplot(2,2,4)
#plt.plot( df_mqtt_5g['txpck/sec'] , 'r--' , label = "5G - MQTT")
plt.plot( df_mqtt_wifi['txpck/sec'] , 'b--' , label = "WiFi - MQTT")
#plt.plot( df_ros_5g['txpck/sec'] , 'g--' , label = "5G - ROS")
plt.plot( df_ros_wifi['txpck/sec'] , 'y--' , label = "WiFi - ROS")
plt.legend()
plt.xlabel("Seconds")
plt.ylabel("txpck/s")

plt.suptitle("Network Throughput (Server)")
plt.savefig('./throughput_results.pdf')

plt.show()
