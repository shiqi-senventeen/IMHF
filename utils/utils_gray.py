import os
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def norm_zscore(tx):
    tx = np.array(tx)
    tx = tx.astype(np.float32)
    tx_flat = tx.flatten()
    if np.sum(tx_flat) > 0:
        tx_flat_no = tx_flat[tx_flat > 0]
        tx_normal = (tx - np.mean(tx_flat_no)) / (np.std(tx_flat_no) + 1e-5)
        tx_normal[tx == 0] = 0
    else:
        tx_normal = tx
    return tx_normal


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    确保增强后的图像和 mask 仍然匹配
    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    参数：
        crop：一个元组，描述随机裁剪的大小。如果 bool(crop) 返回 False，则不会进行裁剪。
        p_flip：浮点数，表示随机水平翻转的概率。
        color_jitter_params：一个元组，描述 torchvision.transforms.ColorJitter 的参数。如果 bool(color_jitter_params) 返回 False，则不使用颜色扰动变换。
        p_random_affine：浮点数，表示随机使用 torchvision.transforms.RandomAffine 的概率。
        long_mask：布尔值，如果为 True，则返回掩码作为标签编码格式的 LongTensor。
    """

    def __init__(self, img_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, z_score=False, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.zscore = z_score
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        # 设置光线抖动范围 ±0.1
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        #  伽马变换
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # 转换为 PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # 随机裁剪
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), 2), F.resize(mask, (new_h, new_w), 0)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        if self.zscore:
            image = norm_zscore(image)
            image = torch.from_numpy(image[None, :, :])
        else:
            image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)
        return image, mask


class ImageToImage2D(Dataset):
    """
    定义了一个数据集类
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.

        joint_transform：一个JointTransform2D实例，这是应用于图像和掩码的增强变换。如果joint_transform为 False，
        则使用torchvision.transforms.ToTensor对图像和掩码进行转换。

        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, split='train1', joint_transform: Callable = None, classes=2,
                 one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'img')
        self.label_path = os.path.join(dataset_path, 'label')
        self.classes = classes
        self.one_hot_mask = one_hot_mask
        id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    # 重写getitem方法
    def __getitem__(self, i):
        id_ = self.ids[i]
        image = cv2.imread(os.path.join(self.img_path, id_ + '.png'), 0)
        mask = cv2.imread(os.path.join(self.label_path, id_ + '.png'), 0)
        # 这一步返回的应该是numpy数组
        # correct dimensions if needed
        # 纠正输入维度，灰度图or RGB
        image, mask = correct_dims(image, mask)
        # 如果是个二分类的问题，将mask里面像素值大于1 的地方替换成1，此时的numpy是一个二维矩阵
        # if self.classes == 2:
        #     mask[mask > 1] = 1
        # 调用joint_transform对image变换
        # if self.joint_transform:
        #     image, mask = self.joint_transform(image, mask)
        #
        # if self.one_hot_mask:
        #     assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
        #     mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        return image, mask, id_ + '.png'


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:

        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
    """

    def __init__(self, dataset_path: str, split='test', transform: Callable = None):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'img')
        self.label_path = os.path.join(dataset_path, 'label')
        id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        image = cv2.imread(os.path.join(self.img_path, id_ + '.png'), 0)
        image = correct_dims(image)
        image = self.transform(image)
        return image, id_ + '.png'


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}
