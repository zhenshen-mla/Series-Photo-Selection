from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import make_shuffle_path


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img


def transform(sample):
    composed_transforms = transforms.Compose([
        FixScaleCrop(crop_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


class MyDataset(data.Dataset):

    def __init__(self, train=True):
        if train:
            self.pathA, self.pathB, self.result, _, _, _ = make_shuffle_path()
        else:
            _, _, _, self.pathA, self.pathB, self.result = make_shuffle_path()
        self.train = train

    def __getitem__(self, index):
        imageA = Image.open('/home/user/train_val_imgs/' + self.pathA[index]).convert('RGB')
        # imageA.show()
        imageB = Image.open('/home/user/train_val_imgs/' + self.pathB[index]).convert('RGB')
        # imageB.show()
        # print(self.result[index])
        if self.train:
            imageA = transform(imageA)
            imageB = transform(imageB)
            return imageA, imageB, int(self.result[index])
        else:
            imageA = transform(imageA)
            imageB = transform(imageB)
            return imageA, imageB, int(self.result[index])

    def __len__(self):
        return len(self.result)


def make_loader():
    train_data = MyDataset(train=True)
    val_data = MyDataset(train=False)
    trainloader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    valloader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    return train_data, val_data, trainloader, valloader