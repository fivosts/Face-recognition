"""
Author: Foivos Tsimpourlas
"""
import subprocess
import json
import pickle
import queue
import socketserver
import socket
import multiprocessing
import requests
import flask
import time
import copy
import waitress
import pathlib
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cnn_stream.util import log as l
from cnn_stream.util import database

app = flask.Flask(__name__)

class FlaskHandler(object):
  def __init__(self):
    self.out_queue = None
    self.backlog = None
    self.most_recent = None
    self.db_path = None
    self.db = None
    self.flush_count = None
    self.flush_queue = None
    return

  def set_queues(self, out_queue: queue.Queue) -> None:
    self.out_queue = out_queue
    self.backlog = []
    self.db_path = pathlib.Path("database/backend.db").absolute()
    self.db_path.parent.mkdir(exist_ok = True, parents = True)
    self.db = database.ImageDatabase("sqlite:///{}".format(str(self.db_path)))
    self.flush_count = 0
    self.flush_queue = []
    return

handler = FlaskHandler()

@app.route('/add_frame', methods=['PUT'])
def add_frame():
  """
  Collect a json file, serialize and place into queue for NN prediction.

  Example command:
    curl -X POST http://localhost:PORT/json_file \
         --header "Content-Type: application/json" \
         -d @/home/fivosts/AITrace_prvt/ML/AITrace/backend/test.json
  """
  byteArray = flask.request.get_data()
  img = pickle.loads(byteArray)

  if not isinstance(img, np.ndarray):
    return "Format Error: Provided instance is not <np.array> but {}\n".format(type(img)), 400
  if len(img.shape) != 3:
    return "Image array is not 3-dimensional. Are you sure this is a valid image ? Dimensions: {}\n".format(len(img.shape)), 400
  handler.out_queue.put(img)
  handler.most_recent = img
  handler.flush_count += 1

  if len(handler.flush_queue) > 0:
    with handler.db.Session() as s:
      for fr in handler.flush_queue:
        s.add(
          database.Image(**database.Image.FromArray(pickle.dumps(fr)))
        )
      s.commit()
    handler.flush_queue = []

  if handler.flush_count > 20:
    with handler.db.Session() as s:
      while not handler.out_queue.empty():
        s.add(
          database.Image(**database.Image.FromArray(pickle.dumps(handler.out_queue.get(block = True))))
        )
      s.commit()
    handler.flush_count = 0
  return 'OK\n', 200

@app.route('/most_recent', methods = ['GET'])
def most_recent():
  """
  Publish all the predicted results of the out_queue.
  Before flushing the out_queue, save them into the backlog.

  Example command:
    curl -X GET http://localhost:PORT/get_predictions
  """
  if handler.most_recent is not None:
    file_object = io.BytesIO()
    img = Image.fromarray(handler.most_recent.astype('uint8'))
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')
    data = {
      'img': base64img
    }
    return flask.render_template("most_recent.html", data = data)
  else:
    return flask.render_template("error.html", data = {'msg': "There have not been received any pictures yet. Wait!"})

def gen():
  while True:
    frame = handler.out_queue.get()
    handler.flush_queue.append(frame)

    file_object = io.BytesIO()
    img = Image.fromarray(frame.astype('uint8'))
    img.save(file_object, 'PNG')
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + file_object.getvalue() + b'\r\n')
  return

@app.route('/video_feed')
def video_feed():
  return flask.Response(gen(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

def hist():
  for dp in handler.db.get_data:
    frame = pickle.loads(dp.image)
    file_object = io.BytesIO()
    img = Image.fromarray(frame.astype('uint8'))
    img.save(file_object, 'PNG')
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + file_object.getvalue() + b'\r\n')

@app.route('/show_history')
def show_history():
  return flask.Response(hist(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit', methods=['POST'])
def submit():
  """
  Submit paragraph prompt.
  """
  action = ""
  for x in flask.request.form:
    if x == "Show most recent" or "Live video feed" or "Show session history":
      action = x
      break
  try:
    if action == "Show most recent":
      return most_recent()
    elif action == "Live video feed":
      return video_feed()
    elif action == "Show session history":
      return show_history()
    else:
      raise ValueError("Unrecognized action")
  except Exception as e:
    return "{}\nThe above exception has occured. Copy paste to Foivos.".format(repr(e)), 404

@app.route('/')
def index():
  """
  Main page of graphical environment.
  """
  return flask.render_template("index.html")

def serve(ip_address: str = "localhost", port: int = 40822) -> None:
  try:
    handler.set_queues(queue.Queue())
    l.logger().info("HTTP server is up at http://{}:{}".format(ip_address, port))
    waitress.serve(app, host = ip_address, port = port)
  except KeyboardInterrupt:
    return
  except Exception as e:
    raise e
  return

def publish_outputs(out_queue: queue.Queue,
                    http_address: str = "localhost",
                    http_port: int = 40822
                    ) -> None:
  """
  Call requests here to post to server.
  """
  while True:
    batch = out_queue.get(block = True)
    try:
      r = requests.put(
        "http://{}:{}/add_frame".format(http_address, http_port),
        data = batch,
        headers = {"Content-Type": "application/json"}
      )
    except Exception as e:
      l.logger().error("PUT Request at {}:{} has failed. Maybe the server is down.".format(http_address, http_port))
      l.logger().error(e)
      time.sleep(2)
    if r.status_code != 200:
      l.logger().error("Error code {} in add_frame request.".format(r.status_code))
      time.sleep(2)
  return

def main():
  l.initLogger("backend_server")
  serve()
  return

if __name__ == "__main__":
  main()
  exit(0)