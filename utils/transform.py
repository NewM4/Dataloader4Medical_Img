import torch
import numpy as np
import torchvision.transforms as transforms
import random


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, img, mask):
        img = torch.tensor(data=img, dtype=torch.float32)
        mask = torch.tensor(data=mask, dtype=torch.float32)

        return img, mask


class Resize(object):
    def __init__(self, sizes):
        """
        :param sizes: (list) 要resize为的分辨率，w * h。
        """
        super(Resize, self).__init__()
        self.sizes = sizes

    def __call__(self, img, mask):
        resize = transforms.Resize(self.sizes)
        temp1, temp2 = resize(img), resize(mask)
        return temp1, temp2


class RandomCrop(object):
    def __init__(self, slices):
        self.slices = slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):
        # print(f"Mask slices is {mask.size(1)}.")
        ss, es = self._get_range(mask.size(1), self.slices)

        # print(f"裁剪中：{img[:, ss:es].size()}, {mask[:, ss:es].size()}")

        return img[:, ss:es], mask[:, ss:es]


class RandomFlip_Slice(object):
    """
    从切片维度上逆置
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _flip(self, img, prob):
        if prob < self.prob:
            img = img.flip(1)
        return img

    def __call__(self, img, masks):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob[0]), self._flip(masks, prob[1])


class RandomFlip_Image(object):
    """
    从单张Image的长宽维度上进行逆置
    """
    def __init__(self, prob=0.5, height=True):
        """
        :param prob: 逆置概率
        :param height: 翻转长或者宽，长为True, 宽为False
        """
        super().__init__()
        self.prob = prob
        self.height = height

    def _flip(self, img, prob):
        if prob < self.prob:
            if self.height == True:
                img = img.flip(2)
            else:
                img = img.flip(3)
        return img

    def __call__(self, img, masks):
        prob =(random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob[0]), self._flip(masks, prob[1])



class Compose(object):
    def __init__(self, trans):
        super(Compose, self).__init__()
        self.trans = trans

    def __call__(self, img, masks):
        for t in self.trans:
            img, masks = t(img, masks)
        return img, masks

if __name__ == "__main__":
    from PIL import Image

    img = np.random.randn(170, 522, 489)
    mask = np.random.randn(170, 522, 489)
    compose = Compose([
        ToTensor(),
        Resize(),
        RandomCrop(48),
    ])
    a = compose(img, mask)
    pass