import os
import cv2
from tqdm import tqdm

image_names = os.listdir('jpg')

for image_name in tqdm(image_names):
	image = cv2.imread(f'jpg/{image_name}')
	w, h, _ = image.shape

	if w > h:
		image = image[w//2-h//2:w//2+h//2, :]
	else:
		image = image[:, h//2-w//2:h//2+w//2]

	image = cv2.resize(image, (224, 224))

	cv2.imwrite(f'images/{image_name}', image)
