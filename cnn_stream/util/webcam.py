"""
Author: Foivos Tsimpourlas
"""
import pickle
import time
import cv2
import numpy as np
from typing import Iterator
import paho.mqtt.publish as publish

from cnn_stream.util import log as l

def stream_webcam() -> Iterator[np.array]:
  """
  Open webcam connection and yield a stream of images from webcam.
  """
  vidcap = cv2.VideoCapture(-1)
  #check if connection with camera is successfully
  if vidcap.isOpened():
    while True:
      success, frame = vidcap.read()  #capture a frame from live video
      #check whether frame is successfully captured
      if success:
        yield frame
      else:
        l.logger().error("Picture was not captured.")
        return
  else:
    l.logger().error("Camera is not open.")
    return

def pub() -> None:
  """
  Send the image.
  """
  MQTT_SERVER = "localhost"
  MQTT_PATH = "Image"

  for img in stream_webcam():
    byteArr = pickle.dumps(img)
    publish.single(MQTT_PATH, byteArr, hostname = MQTT_SERVER)
    ## Some frames may go missing here if the subscribers don't catch them.
  return

def main():
  l.initLogger("webcam_streamer")
  pub()

if __name__ == "__main__":
  main()
  exit(0)