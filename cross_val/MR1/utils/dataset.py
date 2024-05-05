import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def Normalized(img):#标准化
    transform = transforms.Compose([transforms.ToTensor()])
    img_tr = transform(img)
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
    transform_norm = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    img_normalized = transform_norm(img)
    return img_normalized

class Train_load(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, f'with_artifact/*.png'))
        # print(os.path.join(data_path, 'image\*.png'))

    def augment(self, image, angle_range=20, translate_range=20, scale_range=0.1, shear_range=0.1):
        # 随机旋转
        angle = random.uniform(-angle_range, angle_range)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

        # 随机平移
        dx = random.uniform(-translate_range, translate_range)
        dy = random.uniform(-translate_range, translate_range)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, M, (cols, rows))

        # 随机缩放
        scale = random.uniform(1 - scale_range, 1 + scale_range)
        image = cv2.resize(image, None, fx=scale, fy=scale)

        # 随机剪切
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pt1 = 50 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 200 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 50], [pt2, pt1], [50, pt2]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows))

        return image

    def normalized(self, image):
        image_normalized = np.zeros(image.shape, dtype=np.float32)
        cv2.normalize(image, image_normalized, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
        return image_normalized

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('with_artifact', 'no_artifact')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)#260x260,3通道
        label = cv2.imread(label_path)
        image = cv2.resize(image,(256,256))
        label = cv2.resize(label,(256,256))
        # 数据增强
        # image = self.augment(image)
        # label = self.augment(label)
        # 归一化
        image = self.normalized(image)
        label = self.normalized(label)
        # #标准化
        # image = Normalized(image)
        # label = Normalized(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])#reshape为[3,260,260]
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label
    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

class val_load(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'with_artifact/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def normalized(self, image):
        image_normalized = np.zeros(image.shape, dtype=np.float32)
        cv2.normalize(image, image_normalized, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
        return image_normalized

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('with_artifact', 'no_artifact')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)  # 260x260,3通道
        label = cv2.imread(label_path)

        # image=image.numpy()
        image = cv2.resize(image,(256,256))  # 260x260,3通道
        label = cv2.resize(label,(256,256))

        #归一化
        image = self.normalized(image)
        label = self.normalized(label)###归一化后图像质量受损

        #标准化
        # image = Normalized(image)
        # label = Normalized(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])  # reshape为[3,260,260]
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

class test_load(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'with_artifact/*.png'))

    def normalized(self, image):
        image_normalized = np.zeros(image.shape, dtype=np.float32)
        cv2.normalize(image, image_normalized, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
        return image_normalized

    def __getitem__(self, index):
        image_path = self.imgs_path[index]        # 根据index读取图片
        label_path = image_path.replace('with_artifact', 'no_artifact')        # 根据image_path生成label_path
        image = cv2.imread(image_path)  # 260x260,3通道        # 读取训练图片和标签图片
        label = cv2.imread(label_path)
        image = cv2.resize(image,(256,256))  # 260x260,3通道    #改变大小
        label = cv2.resize(label,(256,256))
        image = self.normalized(image)
        label = self.normalized(label)###归一化后图像质量受损            #归一化
        # image = Normalized(image)        #标准化
        # label = Normalized(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)           #转为单通道
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])  # reshape为[1,260,260]
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

def save_deartif_image(image_deartif, name_path):
    image_deartif = image_deartif.view(image_deartif.size(0), 3, 256, 256)
    save_image(image_deartif[0,:,:,:], name_path)

def load_tensor(path,fold):
    train_path = os.path.join(path,f'fold_{fold}/train')
    val_path = os.path.join(path,f'fold_{fold}/val')
    test_path = os.path.join(path,f'fold_{fold}/test')
    train_dataset = Train_load(train_path)
    val_dataset = val_load(val_path)
    test_dataset = test_load(test_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("训练集数据个数：", len(train_dataset))
    print("验证集个数：",len(val_dataset))
    print('测试集个数：',len(test_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=16,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=16,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    # for i, (image, label) in enumerate(train_loader):##可视化预处理过程
    #     image = image.to(device=device, dtype=torch.float32)
    #     print('预处理后的图像：')
    #     print(image.shape)
    #     toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    #     pic = toPIL(image[1,:,:,:])
    #     pic1 = toPIL(label[1,:,:,:])
    #     pic.save('../data1/image.jpg')
    #     pic1.save('../data1/label.jpg')
    print("load data finished")
    # for i,data in enumerate(train_loader):
    #     image = data[0]
    #     print(image.shape)


if __name__ == "__main__":
    path = '../data'
    fold = 1
    # load_tensor(path,1)





