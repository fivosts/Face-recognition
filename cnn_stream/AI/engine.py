"""
Author: Foivos Tsimpourlas
"""
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image 
import pathlib
from typing import Iterator, Tuple, List, Callable, Union

import torch
import torchvision.transforms as T
from torchvision.models import detection
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image

from cnn_stream.AI import example
from cnn_stream.AI import quantizer

dev = "cuda" if torch.cuda.is_available() else "cpu"

class VideoCapture(object):
  """
  This class is an abstract API that blends actual OpenCV videos
  with stand-alone image frames as a list.
  """
  @classmethod
  def FromData(cls, v: Union[str, List[np.array]]) -> 'VideoCapture':
    if isinstance(v, str):
      return cv2.VideoCapture(v)
    else:
      return VideoCapture(v)

  def __init__(self, v: List[np.array]):
    self.CAP_PROP_FRAME_HEIGHT = v[0].shape[0]
    self.CAP_PROP_FRAME_WIDTH = v[0].shape[1]
    self._video = v
    self._size = len(self._video)
    return

  def __len__(self) -> int:
    return len(self._video)

  def read(self) -> Tuple[bool, np.array]:
    """
    Stream sequence of frames as a video.
    """
    if self._size > 0:
      self._size -= 1
      return True, self._video.pop(0)
    return False, None

  def get(self, arg) -> int:
    """
    Get class property. Raises KeyError if not found.
    """
    return {
      cv2.CAP_PROP_FRAME_WIDTH: self.CAP_PROP_FRAME_WIDTH,
      cv2.CAP_PROP_FRAME_HEIGHT: self.CAP_PROP_FRAME_HEIGHT,
    }[arg]


def setup_cnn(quantize_fn: Callable = None) -> torch.nn.Module:
  """
  Initialize torchvision module, place to device and return.

  Returns:
    ResNet50 to device.
  """
  dev = "cuda" if torch.cuda.is_available() else "cpu"
  model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
  if quantize_fn:
    model = quantize_fn(model)
  model = model.to(dev)
  model.eval()
  return model

def run_engine(vid: VideoCapture,
               batch_size: int = 1,
               quantize_fn: Callable = None,
               W: int = None,
               H: int = None,
               model: torch.nn.Module = None
               ) -> Iterator[Tuple[int, np.array, List[np.array]]]:
  """
  !!!# CNN Inference core function.

  Stream a list of predicted faces from an input video.

  Args:
    vpath: Path to input video.
    batch_size: Sets the batch size for model inference. Optimizes throughput but makes single datapoint availability slower.
    quantize_fn: Callable that quantizes the pre-trained CNN.
    W [Optional]: Set to downsample video to specified width.
    H [Optional]: Set to downsample vide to specified height.
  
  Returns:
    An iterator of Tuples. Each tuple has the frame id, boxed image and a list of predicted box coordinates.

  Raises:
    FileNotFoundError:
      If video path does not exist.
    ValueError:
      If there is mismatch in the video's resolution.
  """
  # Get original resolution of video.
  width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

  # Download pre-trained weights and load to device.
  if model is None:
    model = setup_cnn(quantize_fn = quantize_fn)

  # Update any potential downscaled dimensions.
  resize = False
  if W:
    if W > width:
      raise ValueError("Downsample width must not surpass original video resolution.")
    width = W
    resize = True
  if H:
    if H > height:
      raise ValueError("Downsample height must not surpass original video resolution.")
    height = H
    resize = True

  with torch.no_grad():
    success = True
    while success:
      c, img_batch = 0, []
      while c < batch_size and success:
        success, img = vid.read()
        if success:
          if resize:
            img = cv2.resize(img, (width, height))
          img_batch.append(img)
        c += 1
      if len(img_batch) == 0:
        # Not a single new image was fetched, so break.
        break
      # Resize if needed and convert image to batch for model.
      batch = [example.transforms(img) for img in img_batch]
      input_ids = torch.stack(batch)
      if quantize_fn:
        input_ids = quantize_fn(input_ids)

      predictions = model(input_ids.to(dev))
      boxes = [
        [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in prediction['boxes'].detach().cpu()]
        for prediction in predictions
      ]
      for idx, prediction in enumerate(predictions):
        for i in range(len(boxes[idx])):
          cv2.rectangle(img_batch[idx], boxes[idx][i][0], boxes[idx][i][1], color = 500, thickness = 1)
      yield img_batch, boxes

def stream_video(vpath: pathlib.Path,
                 model: torch.nn.Module = None,
                 batch_size: int = 1,
                 quantize_fn: Callable = None,
                 W: int = None,
                 H: int = None,
                 ) -> Iterator[Tuple[int, np.array, List[np.array]]]:
  """
  Predict frames of a .mp4 video provided as a path.
  Load video from path and call run_engine.
  """
  # Video path is not found
  if not vpath.exists():
    raise FileNotFoundError(str(vpath))
  if batch_size < 1:
    raise ValueError("Invalid batch size")
  vid = VideoCapture.FromData(str(vpath))
  return run_engine(vid, batch_size, quantize_fn, W, H, model)

def stream_batch_frames(model: torch.nn.Module,
                        frames: List[np.array],
                        batch_size: int = 1,
                        quantize_fn: Callable = None,
                        W: int = None,
                        H: int = None,
                        ) -> Iterator[Tuple[int, np.array, List[np.array]]]:
  """
  Stream a sequence of frames represented as a video.
  Use this function for online webcam streaming.
  """
  vid = VideoCapture.FromData(frames)
  return run_engine(vid, batch_size, quantize_fn, W, H, model)
