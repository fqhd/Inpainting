import torch
from torch import nn
from constants import *
from modules import *

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.project = nn.Linear(Z_DIM, IMG_SIZE*IMG_SIZE, bias=False)

		kernel_size = 3
		dropout = 0.1
		batch_norm = True
		n_features = 16

		self.pool = nn.MaxPool2d(2, 2)

		self.c128 = ConvBlock(4, n_features, kernel_size, dropout, batch_norm)
		self.c64 = ConvBlock(n_features, n_features * 2, kernel_size, dropout, batch_norm)
		self.c32 = ConvBlock(n_features * 2, n_features * 4, kernel_size, dropout, batch_norm)
		self.c16 = ConvBlock(n_features * 4, n_features * 8, kernel_size, dropout, batch_norm)
		self.c8 = ConvBlock(n_features * 8, n_features * 16, kernel_size, dropout, batch_norm)

		self.u16 = ConvBlock(n_features * 16, n_features * 8, kernel_size, dropout, batch_norm)
		self.u32 = ConvBlock(n_features * 8, n_features * 4, kernel_size, dropout, batch_norm)
		self.u64 = ConvBlock(n_features * 4, n_features * 2, kernel_size, dropout, batch_norm)
		self.u128 = ConvBlock(n_features * 2, n_features, kernel_size, dropout, batch_norm)

		self.u1 = nn.ConvTranspose2d(n_features * 16, n_features * 8, 4, 2, 1)
		self.u2 = nn.ConvTranspose2d(n_features * 8, n_features * 4, 4, 2, 1)
		self.u3 = nn.ConvTranspose2d(n_features * 4, n_features * 2, 4, 2, 1)
		self.u4 = nn.ConvTranspose2d(n_features * 2, n_features, 4, 2, 1)

		self.final = nn.Conv2d(n_features, 3, 3, 1, 1)

		
	def forward(self, x, z):
		b_size = x.shape[0]
		cond = self.project(z)
		cond = torch.reshape(cond, (b_size, 1, IMG_SIZE, IMG_SIZE))
		x = torch.concat((cond, x), dim=1)

		c1 = self.c128(x)
		c2 = self.c64(self.pool(c1))
		c3 = self.c32(self.pool(c2))
		c4 = self.c16(self.pool(c3))
		c5 = self.c8(self.pool(c4))

		c6 = self.u16(torch.concat((self.u1(c5), c4), dim=1))
		c7 = self.u32(torch.concat((self.u2(c6), c3), dim=1))
		c8 = self.u64(torch.concat((self.u3(c7), c2), dim=1))
		c9 = self.u128(torch.concat((self.u4(c8), c1), dim=1))

		final = self.final(c9)
		return torch.sigmoid(final)

if __name__ == '__main__':
	model = Generator()

	x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
	z = torch.randn(16, Z_DIM)

	trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad
	)

	print('Trainable Params:', trainable_params)

	images = model(x, z)

	print(images.shape)
