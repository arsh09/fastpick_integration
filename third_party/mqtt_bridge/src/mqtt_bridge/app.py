import inject
import paho.mqtt.client as mqtt
import rospy

from .bridge import create_bridge
from .mqtt_client import create_private_path_extractor
from .util import lookup_object

class MqttBridgeNode:

    def __init__(self):

        self._debug = True

        # load parameters
        params = rospy.get_param("~", {})
        self.mqtt_params = params.pop("mqtt", {})
        self.conn_params = self.mqtt_params.pop("connection")
        self.mqtt_private_path = self.mqtt_params.pop("private_path", "")
        self.bridge_params = params.get("bridge", [])

        # create mqtt client
        mqtt_client_factory_name = rospy.get_param(
            "~mqtt_client_factory", ".mqtt_client:default_mqtt_client_factory")
        mqtt_client_factory = lookup_object(mqtt_client_factory_name)
        self.mqtt_client = mqtt_client_factory(self.mqtt_params)
        self.mqtt_client.enable_logger()

        # load serializer and deserializer
        serializer = params.get('serializer', 'json:dumps')
        deserializer = params.get('deserializer', 'json:loads')

        # dependency injection
        self.config = self.create_config(
            self.mqtt_client, serializer, deserializer, self.mqtt_private_path)
        inject.configure(self.config)

        # configure and connect to MQTT broker
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect

        # start MQTT loop
        # register shutdown callback and spin
        rospy.on_shutdown(self.mqtt_client.disconnect)
        rospy.on_shutdown(self.mqtt_client.loop_stop)

        self.first_connection = True
        self.bridges = []


    def create_config(self, mqtt_client, serializer, deserializer, mqtt_private_path):
        if isinstance(serializer, str):
            serializer = lookup_object(serializer)
        if isinstance(deserializer, str):
            deserializer = lookup_object(deserializer)
        private_path_extractor = create_private_path_extractor(mqtt_private_path)
        def config(binder):
            binder.bind('serializer', serializer)
            binder.bind('deserializer', deserializer)
            binder.bind(mqtt.Client, mqtt_client)
            binder.bind('mqtt_private_path_extractor', private_path_extractor)
        return config


    def run(self):
        self.mqtt_client.connect_async(**self.conn_params)

        while not rospy.is_shutdown():
            try:
                self.mqtt_client.loop_forever()
            except Exception as e:
                rospy.logwarn('MQTT loop received exception, retrying: %s' % e)
            rospy.sleep(1)


    def _on_connect(self, client, userdata, flags, response_code):
        if self.first_connection:
            rospy.loginfo('MQTT connected for the first time, setting up bridges')
            for bridge_args in self.bridge_params:
                self.bridges.append(create_bridge(**bridge_args))
            self.first_connection = False
        else:
            rospy.loginfo('MQTT re-connected: %s' % str(response_code))


    def _on_disconnect(self, client, userdata, response_code):
        rospy.loginfo('MQTT disconnected: %s', str(response_code))

    def _on_message(self, client, userdata, mqtt_message):
        rospy.loginfo('MQTT message received: %s, %s, %s' %
            client, userdata, mqtt_message
        )

__all__ = ['MqttBridgeNode']
