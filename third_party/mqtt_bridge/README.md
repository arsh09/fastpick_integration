# mqtt_bridge

mqtt_bridge provides a functionality to bridge between ROS and MQTT, bidirectional.


## Principle

`mqtt_bridge` uses ROS message as its protocol. Messages from ROS are serialized by json (or messagepack) for MQTT, and messages from MQTT are deserialized for ROS topic. So the MQTT messages should be ROS message compatible. (We use `rosbridge_library.internal.message_conversion` for message conversion.)

This limitation can be overcome by defining custom bridge class, though.


## Demo

### Prerequisites

```
$ sudo apt install python3-pip
$ sudo apt install ros-noetic-rosbridge-library
$ sudo apt install mosquitto mosquitto-clients
```

### Install python modules

```bash
$ pip3 install -r requirements.txt
```

### launch node

``` bash
$ roslaunch mqtt_bridge demo.launch
```

Publish to `/ping`,

```
$ rostopic pub /ping std_msgs/Bool "data: true"
```

and see response to `/pong`.

```
$ rostopic echo /pong
data: True
---
```

Publish "hello" to `/echo`,

```
$ rostopic pub /echo std_msgs/String "data: 'hello'"
```

and see response to `/back`.

```
$ rostopic echo /back
data: hello
---
```

You can also see MQTT messages using `mosquitto_sub`

```
$ mosquitto_sub -t '#'
```

## Usage

parameter file (config.yaml):

``` yaml
mqtt:
  client:
    protocol: 4      # MQTTv311
  connection:
    host: localhost
    port: 1883
    keepalive: 60
bridge:
  # ping pong
  - factory: mqtt_bridge.bridge:RosToMqttBridge
    msg_type: std_msgs.msg:Bool
    topic_from: /ping
    topic_to: ping
  - factory: mqtt_bridge.bridge:MqttToRosBridge
    msg_type: std_msgs.msg:Bool
    topic_from: ping
    topic_to: /pong
```

you can use any msg types like `sensor_msgs.msg:Imu`.

launch file:

``` xml
<launch>
  <node name="mqtt_bridge" pkg="mqtt_bridge" type="mqtt_bridge_node.py" output="screen">
    <rosparam file="/path/to/config.yaml" command="load" />
  </node>
</launch>
```


## Configuration

### mqtt

Parameters under `mqtt` section are used for creating paho's `mqtt.Client` and its configuration.

#### subsections

* `client`: used for `mqtt.Client` constructor
* `tls`: used for tls configuration
* `account`: used for username and password configuration
* `message`: used for MQTT message configuration
* `userdata`: used for MQTT userdata configuration
* `will`: used for MQTT's will configuration

See `mqtt_bridge.mqtt_client` for detail.

### mqtt private path

If `mqtt/private_path` parameter is set, leading `~/` in MQTT topic path will be replaced by this value. For example, if `mqtt/pivate_path` is set as "device/001", MQTT path "~/value" will be converted to "device/001/value".

### serializer and deserializer

`mqtt_bridge` uses `msgpack` as a serializer by default. But you can also configure other serializers. For example, if you want to use json for serialization, add following configuration.

``` yaml
serializer: json:dumps
deserializer: json:loads
```

### bridges

You can list ROS <--> MQTT tranfer specifications in following format.

``` yaml
bridge:
  # ping pong
  - factory: mqtt_bridge.bridge:RosToMqttBridge
    msg_type: std_msgs.msg:Bool
    topic_from: /ping
    topic_to: ping
  - factory: mqtt_bridge.bridge:MqttToRosBridge
    msg_type: std_msgs.msg:Bool
    topic_from: ping
    topic_to: /pong
```

* `factory`: bridge class for transfering message from ROS to MQTT, and vise versa.
* `msg_type`: ROS Message type transfering through the bridge.
* `topic_from`: topic incoming from (ROS or MQTT)
* `topic_to`: topic outgoing to (ROS or MQTT)

Also, you can create custom bridge class by inheriting `mqtt_brige.bridge.Bridge`.

## Addtional dynamic bridges and service calls (extension by @marc-hanheide)

See [`demo_params.yaml`](config/demo_params.yaml) for the use and parameters of the additional bridges.

### DynamicBridgeServer (Server-Client use case)

The main principle here is that a server-client architecture is implemented, with the idea that not both ends of the mqtt connection need to be necessarily set up, but that a ROS systems (here seen as clients) can connect via MQTT to another ROS system (the "server"), and initiate the transmission of topics from either side, i.e., a client can initiate a local topic (`from_topic`) to be forwarded to the other side via MQTT to be republished there as another topics (`to_topic`). (Note: `from_topic` and `to_topic` could well be the same, in which case the MQTT connections is completely transparent.) The server in this case does not need to be configured at all (other than running the `DynamicBridgeServer`). The client will use the `PublishBridge` to advertise the local `from_topic` on the server-side as `to_topic`. Likewise, the client can subscribe to a server-side topic using the `SubscribeBridge`. Then the `from_topic` is on the remoter (server) side and is forwarded onto the the local `to_topic`. 

The main benefit here is that in all these cases, the server-side does not need further configuration.

### ROS services

Also via `DynamicBridgeServer` it is now possible to expose ROS service either way. Again, one side can expose a local ROS service via the `LocalServiceProxy`, mapping the `local_server` (e.g. `/roscpp/get_logger`) to a `remote_server`.

Similarly, the `RemoteService` bridge allows to access a service available on the `remote_server`.

## License

This software is released under the MIT License, see LICENSE.txt.
