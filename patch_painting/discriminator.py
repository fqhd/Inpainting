import torch
from torch import nn
from constants import *

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(3, FEATURES_D, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),
			
			nn.Conv2d(FEATURES_D, FEATURES_D*2, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(FEATURES_D*2, FEATURES_D*4, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(FEATURES_D*4, FEATURES_D*8, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(FEATURES_D*8, FEATURES_D*16, 4, 2, 1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(FEATURES_D*16, 1, 4, 1, 0),
			nn.Sigmoid()
		)

	def forward(self, input):
		return self.main(input)
    
if __name__ == '__main__':
	x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)

	model = Discriminator()

	out = model(x)

	print(out.shape)