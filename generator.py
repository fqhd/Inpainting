import torch
from torch import nn
from constants import *

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.project = nn.Linear(Z_DIM, IMG_SIZE*IMG_SIZE, bias=False)
		self.stack = nn.Sequential(
			nn.Conv2d(4, FEATURES_G, 5, 1, 2, bias=False),
			nn.BatchNorm2d(FEATURES_G),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			# 64x64

			nn.Conv2d(FEATURES_G, FEATURES_G*2, 5, 1, 2, bias=False),
			nn.BatchNorm2d(FEATURES_G*2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			# 32x32

			nn.Conv2d(FEATURES_G*2, FEATURES_G*4, 3, 1, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			# 16x16

			nn.Conv2d(FEATURES_G*4, FEATURES_G*8, 3, 1, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*8),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			# 8x8

			nn.Conv2d(FEATURES_G*8, FEATURES_G*16, 3, 1, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*16),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
			# 4x4

			nn.ConvTranspose2d(FEATURES_G*16, FEATURES_G*8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*8),
			nn.ReLU(inplace=True),
			# 8x8

			nn.ConvTranspose2d(FEATURES_G*8, FEATURES_G*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*4),
			nn.ReLU(inplace=True),
			# 16x16

			nn.ConvTranspose2d(FEATURES_G*4, FEATURES_G*2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(FEATURES_G*2),
			nn.ReLU(inplace=True),
			# 32x32

			nn.ConvTranspose2d(FEATURES_G*2, 3, 4, 2, 1, bias=False),
			nn.Sigmoid()
			# 64x64
		)
	
	def forward(self, x, z):
		b_size = x.shape[0]
		cond = self.project(z)
		cond = torch.reshape(cond, (b_size, 1, IMG_SIZE, IMG_SIZE))
		x = torch.concat((cond, x), dim=1)
		return self.stack(x)

if __name__ == '__main__':
	model = Generator()

	x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
	z = torch.randn(16, Z_DIM)

	images = model(x, z)

	print(images.shape)
