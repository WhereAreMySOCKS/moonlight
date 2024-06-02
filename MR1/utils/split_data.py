import os
import shutil

import cv2
import numpy as np
import nibabel as nib
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from model.unet_model import UNet

######将nii.gz转为png格式，并划分为训练集、验证集和测试集
class MRImagePreprocessor:
    def __init__(self, data_with_artifact_paths, data_without_artifact_paths):
        self.data_with_artifact_paths = data_with_artifact_paths
        self.data_without_artifact_paths = data_without_artifact_paths
        self.image_pairs = None

    def load_nii_data(self, file_path):
        nii_data = nib.load(file_path)
        data = nii_data.get_fdata()
        # print(data.shape)
        return data

    def normalize_data(self, data):
        # data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-7)
        data_normalized = np.zeros(data.shape, dtype=np.float32)
        cv2.normalize(data, data_normalized, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return data_normalized

    # 随机生成仿射变换矩阵
    def random_affine_transform(self):
        # 随机生成旋转角度、平移距离、缩放比例和剪切参数
        angle = np.random.uniform(-50, 50)
        scale = np.random.uniform(0.9, 1.0)
        tx = np.random.uniform(-10, 10)
        ty = np.random.uniform(-10, 10)
        shear = np.random.uniform(-0.1, 0.1)

        # 构造仿射变换矩阵
        T = np.float32([[scale * np.cos(np.radians(angle)), scale * np.sin(np.radians(angle)), tx],
                        [-scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), ty]])

        return T

    # 对image和label进行仿射变换
    def apply_affine_transform(self,X, Y, T):
        # 对image进行仿射变换
        X_transformed = cv2.warpAffine(X, T, (X.shape[1], X.shape[0]))

        # 对label进行仿射变换
        Y_transformed = cv2.warpAffine(Y, T, (Y.shape[1], Y.shape[0]))

        return X_transformed, Y_transformed

    def judge_zero(self,image):
        if  np.mean(image) < 0.01:
            zero=0
        else:
            zero=1
        return zero

    def create_image_pairs(self):
        self.image_pairs = []
        for with_path, without_path in zip(self.data_with_artifact_paths, self.data_without_artifact_paths):
            data_with_artifact = self.load_nii_data(with_path)
            data_without_artifact = self.load_nii_data(without_path)

            data_with_artifact = self.normalize_data(data_with_artifact)
            data_without_artifact = self.normalize_data(data_without_artifact)

            # 随机生成仿射变换矩阵
            T = self.random_affine_transform()
            # 对image和label进行仿射变换
            data_with_artifact, data_without_artifact = self.apply_affine_transform(data_with_artifact, data_without_artifact, T)

            for i in range(data_with_artifact.shape[1]):  # 使用y轴切片
                with_artifact_slice = data_with_artifact[:, i, :]
                without_artifact_slice = data_without_artifact[:, i, :]
                zero = self.judge_zero(without_artifact_slice)
                print(np.mean(without_artifact_slice))
                if zero==0:
                    continue
                else:
                    self.image_pairs.append((with_artifact_slice, without_artifact_slice))

    def save_dataset_as_images(self, output_dir, num_folds=5):
        if self.image_pairs is None:
            self.create_image_pairs()

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        for fold in range(num_folds):
            fold_dir = os.path.join(output_dir, f'fold_{fold}')
            os.makedirs(fold_dir)
            train_dir = os.path.join(fold_dir, 'train')
            os.makedirs(train_dir)
            val_dir = os.path.join(fold_dir, 'val')
            os.makedirs(val_dir)
            test_dir = os.path.join(fold_dir,'test')
            os.makedirs(test_dir)
            os.makedirs(os.path.join(train_dir, 'with_artifact'))
            os.makedirs(os.path.join(train_dir, 'no_artifact'))
            os.makedirs(os.path.join(val_dir, 'with_artifact'))
            os.makedirs(os.path.join(val_dir, 'no_artifact'))
            os.makedirs(os.path.join(test_dir, 'with_artifact'))
            os.makedirs(os.path.join(test_dir, 'no_artifact'))

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(self.image_pairs)):
            train_pairs = [self.image_pairs[i] for i in train_index]
            test_pairs = [self.image_pairs[i] for i in test_index]
            print(len(train_pairs),len(test_pairs))
            for i,pair in enumerate(test_pairs):
                current_dir = os.path.join(output_dir,f'fold_{fold}/test/')
                with_artifact_img = Image.fromarray((pair[0] * 255).astype(np.uint8))
                without_artifact_img = Image.fromarray((pair[1] * 255).astype(np.uint8))

                with_artifact_img = ImageOps.exif_transpose(with_artifact_img.rotate(90, expand=True))
                without_artifact_img = ImageOps.exif_transpose(without_artifact_img.rotate(90, expand=True))

                with_artifact_img.save(os.path.join(current_dir, f'with_artifact/{i}.png'))
                without_artifact_img.save(os.path.join(current_dir, f'no_artifact/{i}.png'))

            for i,pair in enumerate(train_pairs):
                if i <= len(train_pairs)*0.2:
                    current_dir = os.path.join(output_dir,f'fold_{fold}/val')
                else:
                    current_dir = os.path.join(output_dir,f'fold_{fold}/train')

                with_artifact_img = Image.fromarray((pair[0] * 255).astype(np.uint8))
                without_artifact_img = Image.fromarray((pair[1] * 255).astype(np.uint8))

                with_artifact_img = ImageOps.exif_transpose(with_artifact_img.rotate(90, expand=True))
                without_artifact_img = ImageOps.exif_transpose(without_artifact_img.rotate(90, expand=True))

                with_artifact_img.save(os.path.join(current_dir, f'with_artifact/{i}.png'))
                without_artifact_img.save(os.path.join(current_dir, f'no_artifact/{i}.png'))

            print("fold", fold, "finished")

            # fold_size = len(self.image_pairs) // num_folds
            # test_start = fold * fold_size
            # test_end = (fold + 1) * fold_size
            # val_start = test_end
            # val_end = val_start + fold_size*4//num_folds
            # print((test_end-test_start),(val_end-val_start))
            #
            # for i, pair in enumerate(self.image_pairs):
            #     if i >= test_start and i < test_end:
            #         current_dir = os.path.join(fold_dir, 'test')
            #     elif i >= val_start and i < val_end:
            #         current_dir = os.path.join(fold_dir,'val')
            #     elif val_end >=len(self.image_pairs) and i <= (val_end-len(self.image_pairs)):
            #         current_dir = os.path.join(fold_dir,'val')
            #     else:
            #         current_dir = os.path.join(fold_dir, 'train')
            #
            #     with_artifact_img = Image.fromarray((pair[0] * 255).astype(np.uint8))
            #     without_artifact_img = Image.fromarray((pair[1] * 255).astype(np.uint8))
            #
            #     with_artifact_img = ImageOps.exif_transpose(with_artifact_img.rotate(90, expand=True))
            #     without_artifact_img = ImageOps.exif_transpose(without_artifact_img.rotate(90, expand=True))
            #
            #     with_artifact_img.save(os.path.join(current_dir, f'with_artifact/{i}.png'))
            #     without_artifact_img.save(os.path.join(current_dir, f'no_artifact/{i}.png'))
            # print("fold",fold,"finished")

    def get_dataset(self):
        if self.image_pairs is None:
            self.create_image_pairs()

        return np.array(self.image_pairs)


# 示例用法
# data_with_artifact_paths = ['D:\Graduation project\MR\data1\\nii.gz\\3_lq.nii.gz']
# data_without_artifact_paths = ['D:\Graduation project\MR\data1\\nii.gz\\3__hq.nii.gz']
data_with_artifact_paths = ['../../data1/nii.gz/3_lq.nii.gz',
                            '../../data1/nii.gz/6_lq.nii.gz'
                            ]
data_without_artifact_paths = ['../../data1/nii.gz/3__hq.nii.gz',
                            '../../data1/nii.gz/6__hq.nii.gz'
                            ]

preprocessor = MRImagePreprocessor(data_with_artifact_paths, data_without_artifact_paths)
preprocessor.save_dataset_as_images('../../data')
dataset = preprocessor.get_dataset()
print(dataset.shape)