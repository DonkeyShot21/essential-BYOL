
import torch
from torchvision import transforms as transform_lib
import cv2
from PIL import Image, ImageOps
import numpy as np


class BYOLDataTransform():

    def __init__(self, crop_size, mean, std, blur_prob=(1.0, 0.1), solarize_prob=(0.0, 0.2)):
        assert len(blur_prob) == 2 and len(solarize_prob) == 2, 'atm only 2 views are supported'
        self.crop_size = crop_size
        self.normalize = transform_lib.Normalize(mean=mean, std=std)
        self.color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.transforms = [self.build_transform(bp, sp) for bp, sp in zip(blur_prob, solarize_prob)]

    def build_transform(self, blur_prob, solarize_prob):
        transforms = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.crop_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.RandomApply([self.color_jitter], p=0.8),
            transform_lib.RandomGrayscale(p=0.2),
            transform_lib.RandomApply([GaussianBlur(kernel_size=23)], p=blur_prob),
            transform_lib.RandomApply([Solarize()], p=solarize_prob),
            transform_lib.ToTensor(),
            self.normalize
        ])
        return transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)