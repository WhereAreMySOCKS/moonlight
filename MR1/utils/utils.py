import glob
import logging
import os
import random
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os


def gen_k_fold_index(k,all_data_path,seed,is_full,used_dataset_name_list):
    train_data_path_list = []
    test_data_path_list = []
    if used_dataset_name_list is None:
        dbtype_list = os.listdir(all_data_path)
    else:
        dbtype_list = used_dataset_name_list
    for dbtype in dbtype_list:
        if  os.path.isfile(os.path.join(all_data_path, dbtype)):
            dbtype_list.remove(dbtype)
    all_data_path = [all_data_path + dbtype for dbtype in dbtype_list]
    imgs_path = [glob.glob(i + '/with_artifact/*.png') for i in all_data_path]
    if is_full:
        logging.info(f'根据全量数据划分数据集，共{k}折')
        imgs_path = [item for sublist in imgs_path for item in sublist]
        random.seed(seed)
        random.shuffle(imgs_path)
        num_samples = len(imgs_path)
        fold_size = num_samples // k
        data_path_list = []
        for i in range(k):
            # 为每个折生成一个图像路径子集
            start_index = i * fold_size
            end_index = start_index + fold_size
            # 确保最后一个折包含剩余的样本
            if i == k - 1:
                end_index = num_samples
            # 切片操作获取当前折的图像路径
            fold_data_path = imgs_path[start_index:end_index]
            data_path_list.append(fold_data_path)

        for j in range(k):
            test_data_path_list.append(data_path_list[j])
            train_data_path_list.append([element for sublist in data_path_list[:j] + data_path_list[j+1:] for element in sublist])
    else:
        logging.info(f'根据数据集index划分，共{len(imgs_path)}折')
        for j in range(len(imgs_path)):
            test_data_path_list.append(imgs_path[j])
            # train_data_path_list = imgs_path - test_data_path_list
            train_data_path_list.append([element for sublist in imgs_path[:j] + imgs_path[j+1:] for element in sublist])

    return train_data_path_list,test_data_path_list




def configure_logging(log_file_path, mod,default_level=logging.INFO,):
    """
    配置日志系统，将日志输出到指定的文件路径，并在控制台输出相同内容。
    日志文件将被覆盖，而不是追加。

    :param log_file_path: 日志文件的保存路径
    :param default_level: 日志的默认级别
    """
    # 创建一个日志记录器实例
    logger = logging.getLogger()
    logger.setLevel(default_level)

    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建一个文件处理器，并设置为写入模式，覆盖现有文件
    file_handler = logging.FileHandler(log_file_path, mode=mod)
    file_handler.setLevel(default_level)

    # 创建一个流处理器（控制台）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(default_level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设置处理器的格式
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 避免日志信息重复打印
    logger.propagate = False


def calculate_psnr(prediction,target, max_val=1.0):
    """
    计算原始图像和重建图像之间的峰值信噪比（PSNR）。

    参数:
    - img (Tensor): 原始图像张量 (N, C, H, W)
    - img_g (Tensor): 重建图像张量 (N, C, H, W)
    - max_val (float): 图像可能的最大像素值

    返回:
    - psnr (float): PSNR值
    """
    # 检查原始图像和失真图像的尺寸是否相同
    if target.shape != prediction.shape:
        raise ValueError("原始和失真图像必须具有相同的尺寸")

    # 将图像转换为numpy数组
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # 计算PSNR
    psnr = compare_psnr(target, prediction, data_range=max_val)

    return psnr



def calculate_ssim(prediction,target, max_val=1.0):
    """
    计算原始图像和重建图像之间的结构相似性指数（SSIM）。

    参数:
    - img (Tensor): 原始图像张量 (N, C, H, W)
    - img_g (Tensor): 重建图像张量 (N, C, H, W)
    - max_val (float): 图像可能的最大像素值

    返回:
    - ssim (float): SSIM值
    """
    # 检查原始图像和失真图像的尺寸是否相同
    if target.shape != prediction.shape:
        raise ValueError("原始和失真图像必须具有相同的尺寸")

    # 将图像转换为numpy格式
    target = target.view(target.shape[-2],-1).cpu().numpy()
    prediction = prediction.view(target.shape[-2],-1).cpu().numpy()


    # 计算SSIM
    ssim= compare_ssim(target,prediction,data_range=max_val)

    return ssim
