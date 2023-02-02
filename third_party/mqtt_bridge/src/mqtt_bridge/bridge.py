from abc import ABCMeta
from typing import Optional, Type, Dict, Union

import inject
import paho.mqtt.client as mqtt
import rospy

from .util import lookup_object, extract_values, populate_instance
from threading  import Condition
from queue import Queue
from uuid import uuid4

from threading import Thread

def create_bridge(factory: Union[str, "Bridge"], msg_type: Union[str, Type[rospy.Message]], topic_from: str,
                  topic_to: str, frequency: Optional[float] = None, **kwargs) -> "Bridge":
    """ generate bridge instance using factory callable and arguments. if `factory` or `meg_type` is provided as string,
     this function will convert it to a corresponding object.
    """
    if isinstance(factory, str):
        factory = lookup_object(factory)
    if not issubclass(factory, Bridge):
        raise ValueError("factory should be Bridge subclass")
    if isinstance(msg_type, str):
        msg_type = lookup_object(msg_type)
    if not issubclass(msg_type, rospy.Message):
        raise TypeError(
            "msg_type should be rospy.Message instance or its string"
            "reprensentation")
    return factory(
        topic_from=topic_from, topic_to=topic_to, msg_type=msg_type, frequency=frequency, **kwargs)


class Bridge(object, metaclass=ABCMeta):
    """ Bridge base class """
    _mqtt_client = inject.attr(mqtt.Client)
    _serialize = inject.attr('serializer')
    _deserialize = inject.attr('deserializer')
    _extract_private_path = inject.attr('mqtt_private_path_extractor')


class DynamicBridgeServer(Bridge):
    u""" Dynamic Bridge Server that serves as the remote end to PublishBridge 
    and SubscribeBridge, as well as the RemoteService. Should always be instantiated if 
    indeed the purpose is bridging between ROS-sides.
    """

    def __init__(self, control_topic="__dynamic_server"):
        self._control_topic = control_topic + '/topic/#'
        self._service_topic = control_topic + '/service/request/#'
        self._register_service_topic = control_topic + '/service/register/#'
        self._mqtt_client.subscribe(self._control_topic, qos=2)
        self._mqtt_client.message_callback_add(self._control_topic, self._callback_mqtt_topic)
        self._mqtt_client.subscribe(self._service_topic, qos=2)
        self._mqtt_client.message_callback_add(self._service_topic, self._callback_mqtt_service)
        self._mqtt_client.subscribe(self._register_service_topic, qos=2)
        self._mqtt_client.message_callback_add(self._register_service_topic, self._register_service)
        self._bridges = set([])
        rospy.loginfo('DynamicBridgeServer started on control topic %s' % control_topic)

    def _callback_mqtt_service(self, client, userdata, mqtt_msg):
        t = Thread(target=self.__callback_mqtt_service, args=(userdata, mqtt_msg))
        t.start()

    def __callback_mqtt_service(self, userdata, mqtt_msg):
        rospy.logdebug("MQTT service call received from {}".format(mqtt_msg.topic))
        msg_dict = self._deserialize(mqtt_msg.payload)
        service_type = lookup_object(msg_dict['type'])
        request_type = lookup_object(msg_dict['type'] + 'Request')
        # create request object
        request = request_type()
        # and populate it
        populate_instance(msg_dict['args'], request)
        response_type = lookup_object(msg_dict['type'] + 'Response')

        # create empty response object
        response = response_type()
        msg_dict['op'] = 'response'

        try:
            rospy.logdebug('waiting for service %s' % msg_dict['service'])
            rospy.wait_for_service(msg_dict['service'], 1)

            service = rospy.ServiceProxy(msg_dict['service'], service_type)
            response = service.call(request)
            msg_dict['response'] = extract_values(response)
        except Exception:
            rospy.logerr("Service %s doesn't exist" % msg_dict['service'])
            msg_dict['response'] = None
        finally:
            payload = bytearray(self._serialize(msg_dict))
            self._mqtt_client.publish(
                topic=msg_dict['response_topic'], payload=payload,
                qos=2, retain=False)

    def _register_service(self, client, userdata, mqtt_msg):

        msg_dict = self._deserialize(mqtt_msg.payload)

        if msg_dict['op'] == 'register':
            rospy.loginfo("register service proxy")
            self._bridges.add(RemoteService(
                **msg_dict['args'])
            )

    def _callback_mqtt_topic(self, client, userdata, mqtt_msg):
        u""" callback from MQTT

        :param mqtt.Client client: MQTT client used in connection
        :param userdata: user defined data
        :param mqtt.MQTTMessage mqtt_msg: MQTT message
        """
        msg_dict = self._deserialize(mqtt_msg.payload)

        def __bridge_exists(args):
            for __bridge in self._bridges:
                if __bridge._topic_from == args['topic_to'] and\
                        __bridge._topic_to == args['topic_from']:
                    return True
            return False

        if msg_dict['op'] == 'mqtt2ros_subscribe':
            if not __bridge_exists(msg_dict['args']):
                rospy.loginfo("forward mqtt topic to ros %s" % (
                    msg_dict['args']))
                self._bridges.add(MqttToRosBridge(
                    **msg_dict['args'])
                )
            else:
                rospy.loginfo("bridge for %s already initialised" % (
                    msg_dict['args']))

        if msg_dict['op'] == 'ros2mqtt_subscribe':
            if not __bridge_exists(msg_dict['args']):
                rospy.loginfo("forward ros topic to mqtt %s" % (
                    msg_dict['args']))
                self._bridges.add(RosToMqttBridge(
                    **msg_dict['args'])
                )
            else:
                rospy.logwarn("bridge for %s already initialised" % (
                    msg_dict['args']))


class RosToMqttBridge(Bridge):
    """ Bridge from ROS topic to MQTT

    :param str topic_from: incoming ROS topic path
    :param str topic_to: outgoing MQTT topic path
    :param class msg_type: subclass of ROS Message
    :param (float|None) frequency: publish frequency
    :param bool latched: retain the last message on the MQTT topic (default: False)
    :param int qos: MQTT quality of service (default: 0, max: 2)
    """

    def __init__(self, topic_from, topic_to, msg_type, frequency=None, latched=False, qos=0):
        self._topic_from = topic_from
        self._topic_to = self._extract_private_path(topic_to)
        self._last_published = rospy.get_time()
        self._interval = 0 if frequency is None else 1.0 / frequency
        self._latched = latched
        self._qos = qos
        if isinstance(msg_type, str):
            msg_type = lookup_object(msg_type)
        if not issubclass(msg_type, rospy.Message):
            raise TypeError(
                "msg_type should be rospy.Message instance or its string"
                "reprensentation")

        rospy.Subscriber(topic_from, msg_type, self._callback_ros)

    def _callback_ros(self, msg: rospy.Message):
        rospy.logdebug("ROS received from {}".format(self._topic_from))
        now = rospy.get_time()
        if now - self._last_published >= self._interval:
            self._publish(msg)
            self._last_published = now

    def _publish(self, msg):
        payload = self._serialize(extract_values(msg))
        self._mqtt_client.publish(
            topic=self._topic_to, payload=payload,
            qos=self._qos, retain=self._latched)


class MqttToRosBridge(Bridge):
    """ Bridge from MQTT to ROS topic

    :param str topic_from: incoming MQTT topic path
    :param str topic_to: outgoing ROS topic path
    :param class msg_type: subclass of ROS Message
    :param (float|None) frequency: publish frequency
    :param int queue_size: ROS publisher's queue size (default: 10)
    :param bool latch: latch the ROS topic (default: False)
    :param int qos: MQTT quality of service (default: 0, max: 2)
    """

    def __init__(self, topic_from, topic_to, msg_type, frequency=None,
                 queue_size=10, latched=False, qos=0):
        self._topic_from = self._extract_private_path(topic_from)
        self._topic_to = topic_to
        if isinstance(msg_type, str):
            msg_type = lookup_object(msg_type)
        if not issubclass(msg_type, rospy.Message):
            raise TypeError(
                "msg_type should be rospy.Message instance or its string"
                "reprensentation")
        self._msg_type = msg_type
        self._queue_size = queue_size
        self._latched = latched
        self._qos = qos
        self._last_published = rospy.get_time()
        self._interval = None if frequency is None else 1.0 / frequency
        # Adding the correct topic to subscribe to
        self._mqtt_client.subscribe(self._topic_from, qos=self._qos)
        self._mqtt_client.message_callback_add(self._topic_from, self._callback_mqtt)
        self._publisher = rospy.Publisher(
            self._topic_to, self._msg_type, queue_size=self._queue_size, latch=self._latched)

    def _callback_mqtt(self, client: mqtt.Client, userdata: Dict, mqtt_msg: mqtt.MQTTMessage):
        """ callback from MQTT """
        rospy.logdebug("MQTT received from {}".format(mqtt_msg.topic))
        now = rospy.get_time()

        if self._interval is None or now - self._last_published >= self._interval:
            try:
                ros_msg = self._create_ros_message(mqtt_msg)
                self._publisher.publish(ros_msg)
                self._last_published = now
            except Exception as e:
                rospy.logerr(e)

    def _create_ros_message(self, mqtt_msg: mqtt.MQTTMessage) -> rospy.Message:
        """ create ROS message from MQTT payload """
        # Hack to enable both, messagepack and json deserialization.
        if self._serialize.__name__ == "packb":
            msg_dict = self._deserialize(mqtt_msg.payload, raw=False)
        else:
            msg_dict = self._deserialize(mqtt_msg.payload)
        return populate_instance(msg_dict, self._msg_type())

class SubscribeBridge(MqttToRosBridge):

    def __init__(self, topic_from, topic_to, msg_type, control_topic="__dynamic_server", frequency=None, latched=False, qos=0):
        self._control_topic = control_topic + '/topic/' + topic_from.replace('/', '_')
        self._mqtt_topic = control_topic + '_DATA_' + (topic_from + "_TO_" + topic_to).replace('/','_')
        super(SubscribeBridge, self).__init__(self._mqtt_topic, topic_to, msg_type, frequency, latched, qos)

        rospy.loginfo('SubscribeBridge: subscribe ROS topic %s to topic %s via MQTT %s' %
            (topic_from, topic_to, self._mqtt_topic)
        )

        cmd = {
            'op': 'ros2mqtt_subscribe',
            'args': {
                'topic_from': topic_from, 
                'topic_to': self._mqtt_topic,
                'msg_type': msg_type,
                'frequency': frequency,
                'latched': latched,
                'qos': qos
            }
        }
        payload = bytearray(self._serialize(cmd))
        self._mqtt_client.publish(
            topic=self._control_topic, payload=payload,
            qos=2, retain=False)


class PublishBridge(RosToMqttBridge):

    def __init__(self, topic_from, topic_to, msg_type, control_topic="__dynamic_server", frequency=None, latched=False, qos=0):
        self._control_topic = control_topic + '/topic/' + topic_to.replace('/', '_')
        self._mqtt_topic = control_topic + '_DATA_' + (topic_from + "_TO_" + topic_to).replace('/','_')
        super(PublishBridge, self).__init__(topic_from, self._mqtt_topic, msg_type, frequency, latched, qos)

        rospy.loginfo('PublishBridge: publish from ROS topic %s to topic %s via MQTT %s' %
            (topic_from, topic_to, self._mqtt_topic)
        )

        cmd = {
            'op': 'mqtt2ros_subscribe',
            'args': {
                'topic_from': self._mqtt_topic,
                'topic_to': topic_to,
                'msg_type': msg_type,
                'frequency': frequency,
                'latched': latched,
                'qos': qos
            }
        }
        payload = bytearray(self._serialize(cmd))
        self._mqtt_client.publish(
            topic=self._control_topic, payload=payload,
            qos=2, retain=False)


class LocalServiceProxy(Bridge):

    def __init__(self, local_server, remote_server, srv_type, control_topic="__remote_server"):
        self._register_service_topic = control_topic + '/service/register/' + (local_server + "_TO_" + remote_server).replace('/','_')

        rospy.loginfo('LocalServiceProxy: offer remote access to ROS service  %s as %s via MQTT' %
            (local_server, remote_server)
        )

        cmd = {
            'op': 'register',
            'args': {
                'local_server': remote_server,
                'remote_server': local_server,
                'srv_type': srv_type,
                'control_topic': control_topic
            }
        }
        payload = bytearray(self._serialize(cmd))
        self._mqtt_client.publish(
            topic=self._register_service_topic, payload=payload,
            qos=2, retain=False)


class RemoteService(Bridge):

    def __init__(self, local_server, remote_server, srv_type, control_topic="__remote_server"):
        self._local_server = local_server
        self._remote_server = remote_server
        self._control_topic = control_topic
        self._mqtt_topic_request = self._control_topic + '/service/request/' + (local_server + "_TO_" + remote_server).replace('/','_')

        self._srv_type_name = srv_type
        self._srv_type = lookup_object(self._srv_type_name)
        try:
            self._serviceproxy = rospy.Service(self._local_server, self._srv_type, self._ros_handler)
        except Exception as e: 
            rospy.logwarn(e)

    def _ros_handler(self, req):
        responses = {}
        lock = Condition()

        def __response_handler(client, userdata, mqtt_msg):
            msg_dict = self._deserialize(mqtt_msg.payload)
            rospy.logdebug('got response for %s' % msg_dict['id'])
            with lock:
                responses[msg_dict['id']] = msg_dict['response']
                lock.notifyAll()

        rospy.logdebug('local service %s called.' % self._local_server)
        # generate a unique ID
        request_id = str(uuid4())
        # build a request to send to the external client
        request_message = {
            "op": "call_service",
            "id": request_id,
            "response_topic": self._control_topic + '/service/response/' + request_id,
            "type": self._srv_type_name,
            "service": self._remote_server,
            "args": extract_values(req)
        }
        # Adding the correct topic to subscribe to
        self._mqtt_client.subscribe(request_message['response_topic'], qos=2)
        self._mqtt_client.message_callback_add(request_message['response_topic'], __response_handler)

        payload = bytearray(self._serialize(request_message))
        self._mqtt_client.publish(
            topic=self._mqtt_topic_request, payload=payload,
            qos=2, retain=False)
    
        # wait for a response
        while not rospy.is_shutdown() and request_id not in responses.keys():
            with lock:
                lock.wait(1)  # check for shutdown every 1 second 

        resp = responses[request_id]
        del responses[request_id]
        self._mqtt_client.unsubscribe(request_message['response_topic'])

        # assemble response object
        response_type = lookup_object(self._srv_type_name+"Response")
        # create response object
        r = response_type()
        # and populate it
    
        if resp is None:
            rospy.logerr('Service Request could not be completed')
            raise rospy.ROSException('Service Request could not be completed')

        populate_instance(resp, r)
        
        return r


__all__ = [
    'create_bridge', 'Bridge', 'RosToMqttBridge', 'MqttToRosBridge', 
    'DynamicBridgeServer', 'SubscribeBridge', 'PublishBridge', 'RemoteService', 'LocalServiceProxy']
