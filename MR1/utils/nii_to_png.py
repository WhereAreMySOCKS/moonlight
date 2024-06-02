import logging
import os
import shutil
import numpy as np
import nibabel as nib
from PIL import Image, ImageOps
# from model.unet_model import UNet

######将nii.gz转为png格式，并划分为训练集、验证集和测试集
class MRImagePreprocessor:
    def __init__(self, data_with_artifact_paths, data_without_artifact_paths):
        self.data_with_artifact_paths = data_with_artifact_paths
        self.data_without_artifact_paths = data_without_artifact_paths
        self.image_pairs = None
        self.nii_path = []

    def load_nii_data(self, file_path):
        nii_data = nib.load(file_path)
        data = nii_data.get_fdata()
        # print(data.shape)
        return data

    def normalize_data(self, data):
        data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-7)
        return data_normalized

    def create_image_pairs(self):
        self.image_pairs = []
        for with_path, without_path in zip(self.data_with_artifact_paths, self.data_without_artifact_paths):
            data_with_artifact = self.load_nii_data(with_path)
            data_without_artifact = self.load_nii_data(without_path)

            data_with_artifact_normalized = self.normalize_data(data_with_artifact)
            data_without_artifact_normalized = self.normalize_data(data_without_artifact)

            for i in range(data_with_artifact_normalized.shape[1]):  # 使用y轴切片
                with_artifact_slice = data_with_artifact_normalized[:, i, :]
                without_artifact_slice = data_without_artifact_normalized[:, i, :]
                self.image_pairs.append((with_artifact_slice, without_artifact_slice))
                self.nii_path.append(with_path.split('/')[-1].replace('.nii.gz',''))

    def save_dataset_as_images(self, output_dir, train_test_split=0.8):
        if self.image_pairs is None:
            self.create_image_pairs()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_dir = os.path.join(output_dir, 'train/')
        test_dir = os.path.join(output_dir, 'val/')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        else:
            shutil.rmtree(train_dir)
            os.makedirs(train_dir)
            os.makedirs(os.path.join(train_dir,'with_artifact'))
            os.makedirs(os.path.join(train_dir, 'no_artifact'))

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        else:
            shutil.rmtree(test_dir)
            os.makedirs(test_dir)
            os.makedirs(os.path.join(test_dir,'with_artifact'))
            os.makedirs(os.path.join(test_dir, 'no_artifact'))

        num_train_samples = int(train_test_split * len(self.image_pairs))
        logging.info(f"{num_train_samples} pictures processed.")

        for i, pair in enumerate(self.image_pairs[:num_train_samples]):
            with_artifact_img = Image.fromarray((pair[0] * 255).astype(np.uint8))
            without_artifact_img = Image.fromarray((pair[1] * 255).astype(np.uint8))

            # 逆时针旋转90度
            with_artifact_img = ImageOps.exif_transpose(with_artifact_img.rotate(90, expand=True))
            without_artifact_img = ImageOps.exif_transpose(without_artifact_img.rotate(90, expand=True))

            # 按照数据集名称分别保存图片
            with_artifact_img_path = train_dir + self.nii_path[i] + '/with_artifact/'
            without_artifact_img_path = train_dir + self.nii_path[i] + '/no_artifact/'
            if not os.path.exists(with_artifact_img_path):
                os.makedirs(with_artifact_img_path)
            if not os.path.exists(without_artifact_img_path):
                os.makedirs(without_artifact_img_path)

            with_artifact_img.save(with_artifact_img_path + f'{i}.png')
            without_artifact_img.save(without_artifact_img_path + f'{i}.png')

        for i, pair in enumerate(self.image_pairs[num_train_samples:]):
            with_artifact_img = Image.fromarray((pair[0] * 255).astype(np.uint8))
            without_artifact_img = Image.fromarray((pair[1] * 255).astype(np.uint8))

            with_artifact_img.save(os.path.join(test_dir, f'with_artifact/{i}.png'))
            without_artifact_img.save(os.path.join(test_dir, f'no_artifact/{i}.png'))

    def get_dataset(self):
        if self.image_pairs is None:
            self.create_image_pairs()

        return np.array(self.image_pairs)


# 示例用法
# data_with_artifact_paths = ['../data1/nii.gz/3_lq.nii.gz','../data1/nii.gz/6_lq.nii.gz',
#                             ]
# data_without_artifact_paths = ['../data1/nii.gz/3__hq.nii.gz','../data1/nii.gz/6__hq.nii.gz',
#                             ]
#
# preprocessor = MRImagePreprocessor(data_with_artifact_paths, data_without_artifact_paths)
# preprocessor.save_dataset_as_images('../data1/png', train_test_split=1)
# dataset = preprocessor.get_dataset()
# print(dataset.shape)
