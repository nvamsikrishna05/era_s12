import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    """Gets instance of train and test transforms"""
    mean = (0.4915, 0.4823, .4468)
    train_transform = A.Compose([
        A.Normalize(mean=mean, std=(0.2470, 0.2435, 0.2616), always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=mean, mask_fill_value=None),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=(0.2470, 0.2435, 0.2616), always_apply=True),
        ToTensorV2()
    ])

    return train_transform, test_transform