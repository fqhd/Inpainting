from torch import nn

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout, batch_norm):
		super().__init__()
		self.stack = []

		self.stack.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))
		self.stack.append(nn.ReLU())

		self.stack.append(nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2))
		if batch_norm:
			self.stack.append(nn.BatchNorm2d(out_channels))
		self.stack.append(nn.ReLU())

		if dropout > 0:
			self.stack.append(nn.Dropout(dropout))

		self.stack = nn.Sequential(*self.stack)
	
	def forward(self, x):
		return self.stack(x)
