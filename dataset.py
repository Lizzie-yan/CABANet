import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import tifffile as tiff
import math

class SLSD(data.Dataset):
    def __init__(self, train=True, dataset='SLSD', crop_szie=None, val_full_img=True):  # false
        super(SLSD, self).__init__()
        self.dataset_dir = '/home/data/disk0/ningfan/CABA/work1-Dataset/SLSD'  # /home/tbd/tdwc/dataset/bs/SLSD'
        self.train = train
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie
        if train:
            self.image_dir = self.dataset_dir + '/imgs'
            self.label_dir = self.dataset_dir + '/labels'
            # txt = os.path.join(self.label_dir, 'train.txt')
        else:
            self.image_dir = self.dataset_dir + '/val'
            self.label_dir = self.dataset_dir + '/val_label'

        image_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.label_dir)
        for image_file, label_file in zip(image_files, label_files):
            img_name = image_file.split('.')[0]
            # label_name = label_file.split('.')[0]
            # assert img_name != label_name
            image = os.path.join(self.image_dir, image_file)
            label = os.path.join(self.label_dir, img_name + '.png')
            # image = tiff.imread(image)
            image = Image.open(image)
            image = np.array(image)
            label = Image.open(label)
            label = np.array(label)
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(img_name)
            else:
                slide_crop(image, self.crop_size, self.images)
                slide_crop(label, self.crop_size, self.labels)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. 从内存中获取原始的 image 和 label (NumPy 数组)
        image = self.images[index]
        label = self.labels[index]

        # --- 在这里添加标签清洗代码 ---
        # 打印原始标签中的唯一值，用于第一次Debug确认问题
        # if index == 0: # 只打印一次即可
        #     print(f"SLSD Dataset: 原始标签唯一值: {np.unique(label)}")

        # 强制将所有非0的像素值都设置为1。
        # 这是一个针对二分类（背景为0，前景为1）的鲁棒操作。
        # 它会处理掉 255 或其他任何意外的噪声值。
        label[label != 0] = 1

        # 打印清洗后的标签唯一值，确认修复成功
        # if index == 0:
        #     print(f"SLSD Dataset: 清洗后标签唯一值: {np.unique(label)}")
        # --- 清洗代码结束 ---
        
        # 2. 将清洗后的 label 和 image 打包
        sample = {'image': image, 'label': label}
        
        # 3. 将包含干净 label 的 sample 送入 transform 管道
        sample = self.transform(sample)
        
        if self.val_full_img and not self.train: # 修正一下逻辑，只在验证时添加name
            sample['name'] = self.names[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)

class SL1(data.Dataset):
    def __init__(self, train=True, dataset='SL1', crop_szie=None, val_full_img=True):  # false
        super(SL1, self).__init__()
        self.dataset_dir = '/home/data/disk0/ningfan/CABA/work1-Dataset/SL1'  
        # /home/tbd/tdwc/dataset/bs/SL1/sea-land-segmentation/data-432
        self.train = train
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie
        if train:
            self.image_dir = self.dataset_dir + '/train'
            self.label_dir = self.dataset_dir + '/train_label'
            # txt = os.path.join(self.label_dir, 'train.txt')
        else:
            self.image_dir = self.dataset_dir + '/val'
            self.label_dir = self.dataset_dir + '/val_label'
            # txt = os.path.join(self.label_dir, 'test.txt')

        # with open(txt, "r") as f:
        #     self.filename_list = f.readlines()
        image_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.label_dir)
        for image_file, label_file in zip(image_files, label_files):
            img_name = image_file.split('.')[0]
            # label_name = label_file.split('.')[0]
            # assert img_name != label_name
            image = os.path.join(self.image_dir, image_file)
            label = os.path.join(self.label_dir, img_name + '.png')
            # image = tiff.imread(image)
            image = Image.open(image)
            image = np.array(image)
            label = Image.open(label)
            label = np.array(label)
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(img_name)
            else:
                slide_crop(image, self.crop_size, self.images)
                slide_crop(label, self.crop_size, self.labels)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. 从内存中获取原始的 image 和 label
        image = self.images[index]
        label = self.labels[index]
        
        # --- 在这里添加标签清洗代码 ---
        # 打印原始标签中的唯一值，用于第一次Debug确认问题
        # if index == 0:
        #     print(f"SL1 Dataset: 原始标签唯一值: {np.unique(label)}")

        # 强制将所有非0的像素值都设置为1
        label[label != 0] = 1

        # 打印清洗后的标签唯一值，确认修复成功
        # if index == 0:
        #     print(f"SL1 Dataset: 清洗后标签唯一值: {np.unique(label)}")
        # --- 清洗代码结束 ---
        
        # 2. 将清洗后的 label 和 image 打包
        sample = {'image': image, 'label': label}

        # 3. 将包含干净 label 的 sample 送入 transform 管道
        sample = self.transform(sample)

        if self.val_full_img and not self.train:
            sample['name'] = self.names[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)


class potsdam(data.Dataset):
    def __init__(self, base_dir='./data/', train=True, dataset='vaihingen', crop_szie=None, val_full_img=False):
        super(potsdam, self).__init__()
        self.dataset_dir = base_dir
        self.train = train
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie
        if train:
            self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
            self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/annotations')
            txt = os.path.join(self.label_dir, 'train.txt')
        else:
            self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
            self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/annotations')
            txt = os.path.join(self.label_dir, 'test.txt')

        with open(txt, "r") as f:
            self.filename_list = f.readlines()
        for filename in self.filename_list:
            image = os.path.join(self.image_dir, filename.strip() + '.tif')
            label = os.path.join(self.label_dir, 'labels/' + filename.strip() + '.png')
            image = tiff.imread(image)
            label = Image.open(label)
            label = np.array(label)
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename.strip())
            else:
                slide_crop(image, self.crop_size, self.images)
                slide_crop(label, self.crop_size, self.labels)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        sample = self.transform(sample)
        if self.val_full_img:
            sample['name'] = self.names[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)


def slide_crop(image, crop_size, image_patches):
    """images shape [h, w, c]"""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    stride_rate = 1.0 / 2.0
    h, w, c = image.shape
    H, W = crop_size
    stride_h = int(H * stride_rate)
    stride_w = int(W * stride_rate)
    assert h >= crop_size[0] and w >= crop_size[1]
    h_grids = int(math.ceil(1.0 * (h - H) / stride_h)) + 1
    w_grids = int(math.ceil(1.0 * (w - W) / stride_w)) + 1
    for idh in range(h_grids):
        for idw in range(w_grids):
            h0 = idh * stride_h
            w0 = idw * stride_w
            h1 = min(h0 + H, h)
            w1 = min(w0 + W, w)
            if h1 == h and w1 != w:
                crop_img = image[h - H:h, w0:w0 + W, :]
            if w1 == w and h1 != h:
                crop_img = image[h0:h0 + H, w - W:w, :]
            if h1 == h and w1 == w:
                crop_img = image[h - H:h, w - W:w, :]
            if w1 != w and h1 != h:
                crop_img = image[h0:h0 + H, w0:w0 + W, :]
            crop_img = crop_img.squeeze()
            image_patches.append(crop_img)


def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([0, 0, 0])
    index = image == 1
    RGB[index] = np.array([128, 0, 0])
    return RGB


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    Potsdam_train = potsdam(train=True, dataset='vaihingen')
    dataloader = DataLoader(Potsdam_train, batch_size=1, shuffle=False, num_workers=1)
    # print(dataloader)

    for ii, sample in enumerate(dataloader):
        im = sample['label'].numpy().astype(np.uint8)
        pic = sample['image'].numpy().astype(np.uint8)
        print(im.shape)
        im = np.squeeze(im, axis=0)
        pic = np.squeeze(pic, axis=0)
        print(im.shape)
        im = np.transpose(im, axes=[1, 2, 0])[:, :, 0:3]
        pic = np.transpose(pic, axes=[1, 2, 0])[:, :, 0:3]
        print(im.shape)
        im = np.squeeze(im, axis=2)
        # print(im)
        im = label_to_RGB(im)
        plt.imshow(pic)
        plt.show()
        plt.imshow(im)
        plt.show()
        if ii == 10:
            break
