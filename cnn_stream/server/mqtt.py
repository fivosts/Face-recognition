"""
Author: Foivos Tsimpourlas
"""
import io
import queue
import pickle
import numpy as np
import paho.mqtt.client as mqtt

from cnn_stream.util import webcam
from cnn_stream.util import log as l

def subscribe(stream_queue: queue.Queue) -> None:
  """
  The CNN-inference server should subscribe,
  then webcam server can publish streamed images.

  Connect, then receive images.
  """
  MQTT_SERVER = "localhost"
  MQTT_PATH = "Image"

  def on_connect(client: mqtt.Client, userdata, flags, rc) -> None:
    l.logger().info("Subscriber has succesfully connected.")
    client.subscribe(MQTT_PATH)
    return

  def on_message(client: mqtt.Client, userdata, msg) -> None:
    img = pickle.loads(msg.payload)
    stream_queue.put(img)
    return

  client = mqtt.Client()
  client.on_connect = on_connect
  client.on_message = on_message
  client.connect(MQTT_SERVER, 1883, 60)

  client.loop_forever()
  return
