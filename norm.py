import numpy as np
from torchvision import transforms
import torchstain
from PIL import Image

class MacenkoNormalizerTransform:
    def __init__(self, target_image_pil):
        target_np = np.array(target_image_pil.convert('RGB'))
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        self.normalizer.fit(target_np, beta = 0.15)

    def __call__(self, img_pil):
        img_np = np.array(img_pil.convert('RGB'))
        try: 
            out = self.normalizer.normalize(I=img_np, stains=False)

        except (np.linalg.LinAlgError, ValueError) as e:
            print(img_pil.filename)

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
