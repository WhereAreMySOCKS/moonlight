import argparse
import os.path
import pickle

from sklearn.model_selection import train_test_split

from model.archs import SwinUNet
from model.unet_model import UNet
import torch.nn as nn
from torch import optim

from utils.nii_to_png import MRImagePreprocessor
from model import config
from model.config import get_config
from model.unext_model import UNext
from utils.utils import *
from utils.dataset import *
import warnings

warnings.filterwarnings("ignore") # 忽略警告信息


def train_model(model, criterion, optimizer, train_loader, val_loader, count, args):
    logging.info(f" ==== No.{count} fold is training... ==== ")
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    model.train()
    for e in range(args.epoch):
        running_loss = []

        for images, labels in train_loader:
            images = images.to(device=args.device, dtype=torch.float32)
            labels = labels.to(device=args.device, dtype=torch.float32)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        epoch_train_loss = np.mean(running_loss)
        if val_loader is None:
            # 如果不使用验证集，使用训练损失作为保存模型的指标（不推荐）
            current_loss = epoch_train_loss
            logging.info(f"Epoch {e + 1}/{args.epoch}, Train Loss: {epoch_train_loss}")
        else:
            # 验证模型并计算验证损失
            current_loss = evaluate(model, args.device, criterion, val_loader)
            logging.info(f"Epoch {e + 1}/{args.epoch}, Train Loss: {epoch_train_loss}, Val Loss: {current_loss}")

        # 保存最佳模型
        if current_loss['loss'] < best_val_loss:
            best_val_loss = current_loss['loss']
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path) # 创建保存模型权重的文件夹
            torch.save(model.state_dict(), args.save_path+f'{type(model).__name__}_fold_{count}.pth') # 更新最佳模型权重


def evaluate(_model, _device, _criterion, _val_loader):
    _model.eval()

    with torch.no_grad():
        metrics = {'loss': [], 'psnr': [], 'ssim': []}
        for i, (image, label) in enumerate(_val_loader):
            image = image.to(device=_device, dtype=torch.float32)
            label = label.to(device=_device, dtype=torch.float32)
            pred = _model(image)
            loss = _criterion(pred, label).item()
            metrics['loss'].append(loss)
            metrics['psnr'].append(calculate_psnr(label, pred).item())
            metrics['ssim'].append(calculate_ssim(label, pred).item())

        # 计算平均值
        metrics['loss'] = np.mean(metrics['loss'])
        metrics['psnr'] = np.mean(metrics['psnr'])
        metrics['ssim'] = np.mean(metrics['ssim'])

    return metrics

def train(args):
    train_data_path,test_data_path = gen_k_fold_index(args.k,args.png_data_path,args.seed,args.is_full,args.used_dataset_name_list)
    pickle.dump(test_data_path, open(args.test_data_path, 'wb')) # 将k折的测试集保存，以便评估时使用
    for count, (train_paths, test_paths) in enumerate(zip(train_data_path, test_data_path), start=1):
        if args.model_type == 0:
            model = UNet(n_channels=1).to(args.device)
        elif  args.model_type == 1:
            model = UNext(input_channels=1, img_size=256).to(args.device)
        else:
            model = SwinUNet(config=get_config(config)).to(args.device)
            model.load_from_config()
        logging.info(f" ==== No.{count} {type(model).__name__} model is initializing ==== ")
        if args.is_val: # 如果使用验证集
            # 可调整分割比例，改变学习样本数量
            train_paths,val_paths = train_test_split(train_paths, test_size=0.1, random_state=arg.seed)
            val_dataset = val_load(os.path.join(args.data_path, f'val'),imgs_path=val_paths)
            val_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=val_dataset, shuffle=True)
            logging.info("validation set is used.")
        else: # 可能会导致过拟合，不推荐
            val_loader = None
            logging.info("No validation set is used.")

        train_dataset = Train_load(os.path.join(args.data_path, f'train'),imgs_path=train_paths)
        train_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=train_dataset, shuffle=True)
        optimizer = optim.Adam(model.parameters(), args.learning_rate)
        criterion = nn.MSELoss()
        train_model(model, criterion, optimizer, train_loader, val_loader, count, args)

        logging.info(f"Training for fold No. {count} has been completed. Test set is {test_paths}")

def metric(args):
    logging.info(" ==== Start evaluating ==== ")
    test_data_path = pickle.load(open(args.test_data_path, 'rb')) # 读取先前保存的测试集
    test_metrics = {'loss': [], 'psnr': [], 'ssim': []}
    for count, test_paths in enumerate(test_data_path, start=1):
        if args.model_type == 0:
            model = UNet(n_channels=1).to(args.device)
        elif args.model_type == 1:
            model = UNext(input_channels=1, img_size=256).to(args.device)
        else:
            model = SwinUNet(config=get_config(config)).to(args.device)
            model.load_from_config() # 加载预训练模型
        model.load_state_dict(torch.load(args.save_path+f'{type(model).__name__}_fold_{count}.pth')) # 加载对应模型参数
        test_dataset = test_load(os.path.join(args.data_path, f'test'),imgs_path=test_paths)
        test_loader = torch.utils.data.DataLoader(batch_size=1, dataset=test_dataset, shuffle=False)
        criterion = nn.MSELoss()
        # 评估模型并获取所有指标
        result_metrics = evaluate(model, args.device, criterion, test_loader)

        # 记录每个折的指标
        logging.info(f"Fold No. {count} metrics: {result_metrics}")

        # 将每个折叠的指标添加到总列表中
        for key in test_metrics.keys():
            test_metrics[key].append(result_metrics[key])

        # 计算并记录所有折叠的平均指标
    for key, values in test_metrics.items():
        mean_value = np.mean(values)
        logging.info(f"{args.k} folds cross validation result: mean {key.upper()}: {mean_value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--log_path', type=str, default="./log/", help='path to save log.')
    parser.add_argument('--png_data_path', type=str, default="./data1/png/train/", help='path to png data.')
    parser.add_argument('--save_path', type=str, default="./model/model_pth/", help='path to save model.')
    parser.add_argument('--data_path', type=str, default="./data1/png", help='path to data.')
    parser.add_argument('--gpu', type=str, default='4', help='gpu device index.')
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--k', type=int, default=5, help='number of folds.')
    parser.add_argument('--model_type', type=int, default=0, help='model type.')
    parser.add_argument('--is_val', type=bool, default=True, help='whether to use validation set.')
    parser.add_argument('--test_data_path', type=str, default="./log/test_data_path.pkl", help='record of test dataset.')
    parser.add_argument('--used_dataset_name_list', type=list, default=['23_lq', '25_lq', '6_lq'], help='used dataset name list.')
    parser.add_argument('--is_full', type=bool, default=False, help='whether to use full dataset.')
    arg = parser.parse_args()
    arg.device = torch.device('cuda:'+arg.gpu) if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    # 根据 arg.model_type 选择模型类型
    if arg.model_type == 0:
        model_type = 'UNet'
    elif arg.model_type == 1:
        model_type = 'UNext'
    else:
        model_type = 'SwinUNet'
    # 构建日志文件的路径
    log_filename = f"{arg.log_path}{model_type}.log"
    configure_logging(log_filename,mod='w')
    # 记录参数信息
    logging.info(f" ==== Using {model_type} model ==== ")
    logging.info(
        f'batch_size: {arg.batch_size}, learning_rate: {arg.learning_rate}, epoch: {arg.epoch}, is_full: {arg.is_full}, 使用数据集: {arg.used_dataset_name_list}')

    logging.info(' ==== Convert nii.gz to png ==== ')
    if os.path.exists(arg.png_data_path):
        logging.info('PNG files already exist.')
    else:
        # 将nii.gz文件转换为png文件并保存
        data_with_artifact_paths = glob.glob('./data1/nii.gz/*_lq.nii.gz')
        data_without_artifact_paths = glob.glob('./data1/nii.gz/*_hq.nii.gz')
        preprocessor = MRImagePreprocessor(sorted(data_with_artifact_paths), sorted(data_without_artifact_paths)) # 排序使名称一一对应
        preprocessor.save_dataset_as_images('./data1/png', train_test_split=1) # 使用k-fold交叉验证时不在此处分割数据集

    train(arg) # 训练模型
    metric(arg) # 评估模型
