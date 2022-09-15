import cv2
import albumentations

def default_aug(img_size):
    return albumentations.Compose([
        albumentations.augmentations.geometric.rotate.SafeRotate((-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.augmentations.geometric.resize.RandomScale((0.9, 1.2), p=0.5),
        albumentations.RandomBrightnessContrast(),
        albumentations.augmentations.geometric.resize.Resize(img_size[0], img_size[1], p=1)
    ], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False))

def fake(**kwargs):
    return {**kwargs}
