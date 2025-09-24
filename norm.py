import numpy as np
from torchvision import transforms
import torchstain

class MacenkoNormalizerTransform:
    def __init__(self, target_image_pil):
        target_np = np.array(target_image_pil.convert('RGB'))
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        self.normalizer.fit(target_np)

    def __call__(self, img_pil):
        img_np = np.array(img_pil.convert('RGB'))
        out = self.normalizer.normalize(I=img_np, stains=False)
        if isinstance(out, tuple):
            out = out[0]
        return transforms.ToPILImage()(out.astype(np.uint8))


class ReinhardNormalizerTransform:
    def __init__(self, target_image_pil):
        target_np = np.array(target_image_pil.convert('RGB'))
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='numpy')
        self.normalizer.fit(target_np)

    def __call__(self, img_pil):
        img_np = np.array(img_pil.convert('RGB'))
        out = self.normalizer.normalize(I=img_np)
        if isinstance(out, tuple):
            out = out[0]
        return transforms.ToPILImage()(out.astype(np.uint8))
