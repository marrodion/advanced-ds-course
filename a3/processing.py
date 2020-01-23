import torchvision.transforms as T
from albumentations.pytorch import ToTensor
import albumentations as alb

def get_transforms(train):
    transforms = []    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_albumenations_tranforms(train):
    # From https://arxiv.org/pdf/1906.11172v1.pdf
    transforms = []
    if train:
        transforms.extend([
            get_aug([alb.Posterize(p=0.8, num_bits=2), alb.ShiftScaleRotate(p=1, rotate_limit=0, scale_limit=0, shift_limit=0.1)]),
            get_aug([alb.CropNearBbox(), alb.Sharpness()]),

        ])
    return get_aug(transforms)


def get_aug(aug, min_area=0., min_visibility=0.):
    bbox_param = BboxParams(format='coco', 
                            min_area=min_area, 
                            min_visibility=min_visibility, 
                            label_fields=['labels'])
    return Compose(aug, bbox_params=bbox_param)
