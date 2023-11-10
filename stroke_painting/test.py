import torch
from dataset import dataloader
import matplotlib.pyplot as plt
import torchvision.transforms as T
from constants import *

PATCH_SIZE = 64

model = torch.load('models/stroke_generator.pkl')

fixed_images, fixed_strokes = next(iter(dataloader))
fixed_strokes = fixed_strokes / 255.0
fixed_strokes = fixed_strokes.reshape(BATCH_SIZE, 1, 128, 128)
fixed_strokes = fixed_strokes.expand_as(fixed_images)

fixed_noise = torch.randn(BATCH_SIZE, Z_DIM)

fixed_images = fixed_images * fixed_strokes

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	img = T.ToPILImage()(fixed_images[i])
	plt.imshow(img)
plt.show()

with torch.no_grad():
	predicted_patch = model(fixed_images, fixed_noise)
fixed_images = fixed_images + predicted_patch * (1.0 - fixed_strokes)

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	img = T.ToPILImage()(fixed_images[i])
	plt.imshow(img)
plt.show()