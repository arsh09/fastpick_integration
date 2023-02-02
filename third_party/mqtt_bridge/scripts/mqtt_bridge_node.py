#!/usr/bin/env python3
import rospy

from uuid import getnode, uuid4
from mqtt_bridge.app import MqttBridgeNode


rospy.init_node('mqtt_bridge_node')

# set the client_id to the mac address + something unique
rospy.set_param(
    '~mqtt/client/client_id',
    (
        rospy.get_name() 
        + '_'
        + ':'.join('%02x' % ((getnode() >> 8*i) & 0xff) for i in reversed(range(6)))
        + '_' + str(uuid4())
    )
)

rospy.set_param(
    '~mqtt/client/clean_session', 
    False
)


node = MqttBridgeNode()

node.run()
