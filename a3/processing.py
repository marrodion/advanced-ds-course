import torchvision.transforms as T

def get_transforms(train):
    transforms = []    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)