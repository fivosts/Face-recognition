import pathlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as T
from torchvision.models import detection
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import read_image

import cv2
import numpy as np
from PIL import Image 

import wget
if not pathlib.Path("data/img1.jpg").absolute().exists():
  pathlib.Path("data").absolute().mkdir(exist_ok = True, parents = True)
  wget.download("https://alk15.github.io/home/files/img1.jpg", out = str(pathlib.Path("data/img1.jpg").absolute()))

# Model download and setup

dev = "cuda" if torch.cuda.is_available() else "cpu"

model = detection.fasterrcnn_resnet50_fpn(pretrained = True)
model = model.to(dev)
model.eval()

transforms = []
transforms.append(T.ToTensor())
#transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
transforms = T.Compose(transforms)

# Prepare Image
x = Image.open('data/img1.jpg').convert("RGB")
x = transforms(x)
x = x.unsqueeze(0) 
x = x.to(dev)

# Run Inference
with torch.no_grad():
    prediction = model(x)[0]

# Process Output
scores = prediction["scores"].cpu().numpy()
# print('Scores:', scores)

boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in prediction['boxes'].detach().cpu()]

# Visualise Result
import matplotlib.pyplot as plt

img = Image.open('data/img1.jpg').convert("RGB")
img = np.array(img)
for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1], color=500, thickness=1)

plt.imshow(img)
