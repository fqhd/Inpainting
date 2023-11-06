import torch
from dataset import dataloader
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T

PATCH_SIZE = 64

model = torch.load('generator.pkl', map_location=torch.device('cpu'))

images = next(iter(dataloader))[0]
images = torch.reshape(images, (1, 3, 128, 128))

x_offset = random.randint(0, PATCH_SIZE)
y_offset = random.randint(0, PATCH_SIZE)

for i in range(200):
	t = i / 200
	images[0, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0
	noise = torch.randn(images.shape[0], 128)
	with torch.no_grad():
		predicted_patches = model(images, noise)
	images[0, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches
	img = T.ToPILImage()(images[0])
	img.save(f'frames/frame_{i}.jpg')
