import torch

BATCH_SIZE = 16
Z_DIM = 100
EPOCHS = 300
BETA_1 = 0.5
BETA_2 = 0.999
LEARNING_RATE = 1e-4
FEATURES_D = 32
IMG_SIZE = 128

device = (
	'cuda' if torch.cuda.is_available()
	else 'mps' if torch.backends.mps.is_available()
	else 'cpu'
)