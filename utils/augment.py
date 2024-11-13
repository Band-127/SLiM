import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = (
    "1"  # Disable updates checking for albumentations
)

import albumentations as A


class DarkAug(object):
    """
    Extreme dark augmentation aiming at Aachen Day-Night
    """

    def __init__(self) -> None:
        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(
                    p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MobileAug(object):
    """
    Random augmentations aiming at images of mobile/handhold devices.
    """

    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.MotionBlur(p=0.25),
                A.ColorJitter(p=0.5),
                A.RandomRain(p=0.1),  # random occlusion
                A.RandomSunFlare(p=0.1),
                A.ImageCompression(p=0.25, compression_type="jpeg"),
                A.ISONoise(p=0.25),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MAFFAug(object):
    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.ColorJitter(
                    p=0.25,
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.5,
                ),
                A.RandomRain(p=0.1),
                A.RandomSunFlare(p=0.1),
                A.RandomFog(p=0.1),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.1, blur_limit=(3, 25)),
                A.ImageCompression(p=0.25, quality_lower=50, quality_upper=80),
                A.ISONoise(p=0.25),
            ]
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MAFFLiteAug(object):
    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.ColorJitter(
                    p=0.2,
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.5,
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.ImageCompression(p=0.1, quality_lower=60, quality_upper=80),
                A.ISONoise(p=0.1),
            ]
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


def build_augmentor(method=None, **kwargs):
    if method == "dark":
        return DarkAug()
    elif method == "mobile":
        return MobileAug()
    elif method == "maff":
        return MAFFAug()
    elif method == "maff_lite":
        return MAFFLiteAug()
    elif method is None:
        return None
    else:
        raise ValueError(f"Invalid augmentation method: {method}")


def get_augmentor_builder(method=None, **kwargs):
    if method == "dark":
        return DarkAug
    elif method == "mobile":
        return MobileAug
    elif method == "maff":
        return MAFFAug    
    elif method == "maff_lite":
        return MAFFLiteAug
    elif method is None:
        return None
    else:
        raise ValueError(f"Invalid augmentation method: {method}")
