"""
Author: Foivos Tsimpourlas
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import tqdm
import time
import pickle
import threading
import queue
import pathlib
import wget
import numpy as np
from typing import Callable, List

from cnn_stream.AI import engine
from cnn_stream.AI import quantizer
from cnn_stream.util import log as l
from cnn_stream.util import plotter as plt
from cnn_stream.server import mqtt
from cnn_stream.server import http_server

url = "https://alk15.github.io/home/files/london-walk.mp4"

def benchmark_engine(W: int = 427, H: int = 240) -> None:
  """
  Calls to CNN inference engine for Task 1, 2 and 3.

  Tries different batch size configurations along with quantization
  and measures the model's performance.

  Args:
    W, H: Set the downscale resolution for the processed video.
  """
  ## Download missing files
  p = pathlib.Path("data/london-walk.mp4").absolute()
  if not p.exists():
    p.parent.mkdir(exist_ok = True, parents = True)
    wget.download(url, out = str(p))

  plot_path = pathlib.Path("plots/").absolute()
  plot_path.mkdir(exist_ok = True, parents = True)

  ## Setup
  batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
  q_fn = [None, quantizer.halve]

  results = {}
  for batch_size in batch_sizes:
    for func in q_fn:
      try:
        l.logger().info("Trying batch size {} with {} quantization function".format(batch_size, "no" if func is None else str(func)))

        setup_name = "b_s_{}".format(batch_size)
        if func is not None:
          setup_name += "_w/_{}".format(str(func))

        model = engine.setup_cnn(quantize_fn = func)
        ## Collect iterator
        it = engine.stream_video(p, quantize_fn = func, batch_size = batch_size, W = W, H = H, model = model)
        stream_bar = tqdm.tqdm(desc = "Streaming frames...", leave = False)

        total_elements = 0
        batch_execution_times = []

        try:
          while True:
            t2 = time.time()
            imgs, boxes = next(it)
            t3 = time.time()
            batch_execution_times.append(t3 - t2)
            stream_bar.update(len(imgs))
            total_elements += len(imgs)
        except StopIteration:
          pass

        total_execution_time = sum(batch_execution_times)
        avg_time_per_batch = len(batch_execution_times) / sum(batch_execution_times)
        avg_time_per_element = total_execution_time / total_elements

        results[setup_name] = (
          ['total_time', 'time_per_batch', 'time_per_element'],
          [total_execution_time, avg_time_per_batch, avg_time_per_element]
        )
      except RuntimeError:
        l.logger().warn("Batch size {} too big for GPU. Skipping...".format(batch_size))

  plt.GrouppedBars(
    results,
    plot_name = "CNN_Benchmarking",
    path = plot_path,
  )
  return

def distributed_engine(video_batch_size: int,
                       cnn_batch_size: int,
                       quantize_fn: Callable = None,
                       W: int = 427,
                       H: int = 240,
                       ) -> None:
  """
  Task 4 and 5. The engine subscribes using MQTT and expects published images
  from a separate machine (currently local) to performance inference.
  
  This function uses two daemon threads; one subscribes to new images and the
  other sends predicted ones through http to a backend server.

  Args:
    video_batch_size: The number of frames to collect from webcam before sending to CNN.
    cnn_batch_size: The batch size used for model inference.
    quantize_fn: Callable to one of quantizer.py functions.
    W, H: Set the downscale resolution for the processed video.
  """

  ## Do I need to call stream video ? Maybe create an API to effectively batch collected frames.
  # Initialize required queues.
  webcam_queue = queue.Queue()
  publish_queue = queue.Queue()

  # Start subscriber to read images.
  webcam_thread = threading.Thread(
    target = mqtt.subscribe,
    kwargs = {
      'stream_queue': webcam_queue,
    },
    daemon = True
  )
  publish_thread = threading.Thread(
    target = http_server.publish_outputs,
    kwargs = {
      'out_queue': publish_queue,
    },
    daemon = True,
  )
  webcam_thread.start()
  publish_thread.start()

  # Pre-initialize the model to avoid doing that again and again.
  model = engine.setup_cnn()

  def ExpectInputs() -> List[np.array]:
    """
    Read queue inputs from webcam.
    """
    batch = []
    c = 0
    while c < video_batch_size:
      try:
        cur = webcam_queue.get_nowait()
        batch.append(cur)
        c += 1
      except queue.Empty:
        ## There is no need to block if no frame is available.
        return batch
  try:
    while True:
      if not webcam_thread.is_alive():
        l.logger().error("Webcam thread is dead. Exiting...")
        return
      if not publish_thread.is_alive():
        l.logger().error("Publish server thread is dead. Exiting...")
        return

      batch = ExpectInputs()
      if not batch:
        ## Nothing collected from webcam
        continue
      it = engine.stream_batch_frames(model, batch, cnn_batch_size, quantize_fn = quantize_fn, W = W, H = H)
      for batch in it:
        imgs, boxes = batch
        for img in imgs:
          publish_queue.put(pickle.dumps(img))
  except KeyboardInterrupt:
    l.logger().info("Gracefully exiting...")
  return

def main():
  l.initLogger("cnn_engine")
  if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
    benchmark_engine()
  else:
    distributed_engine(4, 4)

if __name__ == "__main__":
  main()
  exit(0)