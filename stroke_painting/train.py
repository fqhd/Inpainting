from dataset import dataloader
from generator import Generator
from discriminator import Discriminator
from constants import *
from tqdm import tqdm
from torch import nn
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import time

print(f'Started training using device: {device}')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

d_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
fixed_images, fixed_strokes = next(iter(dataloader))

fixed_images = fixed_images.to(device)
fixed_strokes = fixed_strokes.to(device)
fixed_strokes = fixed_strokes / 255.
fixed_strokes = fixed_strokes.reshape(BATCH_SIZE, 1, 128, 128)
fixed_strokes = fixed_strokes.expand_as(fixed_images)

start = time.time()
for epoch in range(EPOCHS):
	for image_batch, mask_batch in tqdm(dataloader):
		image_batch = image_batch.to(device)
		mask_batch = mask_batch.to(device)

		b_size = image_batch.shape[0]

		mask_batch = mask_batch / 255.
		mask_batch = mask_batch.reshape(b_size, 1, 128, 128)
		mask_batch = mask_batch.expand_as(image_batch)

		discriminator.zero_grad()

		y_hat_real = discriminator(image_batch).view(-1)
		y_real = torch.ones_like(y_hat_real, device=device)
		real_loss = loss_fn(y_hat_real, y_real)
		real_loss.backward()

		# Make part of the image black
		image_batch = image_batch * mask_batch

		# Predict using generator
		noise = torch.randn(b_size, Z_DIM, device=device)
		predicted_patch = generator(image_batch, noise)

		# Replace black patch with generator output
		image_batch = image_batch + predicted_patch * (1.0 - mask_batch)

		# Predict fake images using discriminator
		y_hat_fake = discriminator(image_batch.detach()).view(-1)

		# Train discriminator
		y_fake = torch.zeros_like(y_hat_fake)
		fake_loss = loss_fn(y_hat_fake, y_fake)
		fake_loss.backward()
		d_opt.step()

		# Train generator
		generator.zero_grad()
		y_hat_fake = discriminator(image_batch).view(-1)
		g_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
		g_loss.backward()
		g_opt.step()

	fixed_images = fixed_images * fixed_strokes
	with torch.no_grad():
		predicted_patch = generator(fixed_images, fixed_noise)
	fixed_images = fixed_images + predicted_patch * (1.0 - fixed_strokes)
	img = T.ToPILImage()(vutils.make_grid(fixed_images.to('cpu'), normalize=True, padding=2))
	img.save(f'progress/epoch_{epoch}.jpg')
train_time = time.time() - start
print(f'Total training time: {train_time // 60} minutes')


generator = generator.to('cpu')

torch.save(generator, 'models/stroke_generator.pkl')
