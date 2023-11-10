import torch
from dataset import test_ds
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T

PATCH_SIZE = 64

model = torch.load('models/patch_generator.pkl', map_location=torch.device('cpu'))

images = next(iter(test_ds))
images = torch.reshape(images, (16, 3, 128, 128))

x_offset = random.randint(0, PATCH_SIZE)
y_offset = random.randint(0, PATCH_SIZE)

images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(images[i]))
plt.show()

noise = torch.randn(images.shape[0], 128)
with torch.no_grad():
	predicted_patches = model(images, noise)
images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(images[i]))
plt.show()
