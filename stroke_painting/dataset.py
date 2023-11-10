import torch.utils.data as dutils
import os
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import random

class Dataset(dutils.Dataset):
	def __init__(self, transform=None):
		self.image_names = os.listdir('dataset')
		self.strokes = np.load('strokes.npy')
		self.transform = transform

	def __len__(self):
		return len(self.image_names)
	
	def __getitem__(self, index):
		image_name = self.image_names[index]
		image = read_image(f'dataset/{image_name}')
		if self.transform:
			image = self.transform(image)
		stroke_idx = random.randint(0, len(self.strokes) - 1)
		return image, self.strokes[stroke_idx]
	
transform = T.Compose([
	T.ToPILImage(),
	T.RandomRotation(10),
	T.CenterCrop(192),
	T.RandomCrop(180),
	T.Resize(IMG_SIZE),
	T.RandomHorizontalFlip(),
	T.ToTensor()
])

dataset = Dataset(transform=transform)
dataloader = dutils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
	images = next(iter(dataloader))
	plt.figure(figsize=(8, 8))
	plt.title('Images')
	plt.axis('off')
	for i in range(16):
		plt.subplot(4, 4, i+1)
		plt.imshow(T.ToPILImage()(images[i]))
	plt.show()
