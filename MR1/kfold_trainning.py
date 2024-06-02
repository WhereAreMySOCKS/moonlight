import os.path
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model.unet_model import UNet
from model.unext_model import UNext
from model.archs import SwinUNet
from model import config
from model.config import get_config
from early_stop import EarlyStopping
from utils.dataset import Train_load,val_load

def save_deartif_image(image_deartif, name_path):
    image_deartif = image_deartif.view(image_deartif.size(0), 1, 224, 224)
    save_image(image_deartif[:,:,:,:], name_path)

def fit(model, device,train_loader, criterion, optimizer):    #训练集训练代码
    model.train()
    running_loss = []
    for i,(image,label) in enumerate(train_loader):
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)
        # if i%10==0:
        #     save_deartif_image(label.cpu().data,
        #                        f"./pred/label{i}.png")
        #     save_deartif_image(image.cpu().data,
        #                        f"./pred/image{i}.png")
        #     save_deartif_image(pred.cpu().data,
        #                        f"./pred/pred{i}.png")
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    train_loss = np.sum(running_loss) / len(train_loader)
    return train_loss

def validate(model, val_loader, criterion,device):    #验证集预测代码
    model.eval()
    running_loss = []
    running_ssim = []
    running_psnr = []
    with torch.no_grad():
        for i,(image,label) in enumerate(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(image)
            loss = criterion(pred, label)
            pred2 = pred.cpu().detach().numpy()
            label2 = label.cpu().detach().numpy()
            acc = ssim(pred2[0, 0, :, :], label2[0, 0, :, :], data_range=1.0)
            acc2 = psnr(pred2[0, 0, :, :], label2[0, 0, :, :], data_range=1.0)
            running_ssim.append(acc)
            running_psnr.append(acc2)
            running_loss.append(loss.item())
            # save_deartif_image(label.cpu().data,
            #                    f"./pred/label{epoch}.png")
            # save_deartif_image(image.cpu().data,
            #                    f"./pred/image{epoch}.png")
            # save_deartif_image(pred.cpu().data,
            #                     f"./pred/pred{epoch}.png")
        val_loss = np.sum(running_loss) / len(val_loader)
        val_ssim = np.sum(running_ssim)/(len(val_loader))
        val_psnr = np.sum(running_psnr)/(len(val_loader))
        return val_loss,val_psnr,val_ssim

def test(model,criterion,test_loader,device):
    model.eval()
    running_loss = []
    running_mse = []
    running_ssim = []
    running_psnr = []
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(image)
            loss = criterion(pred, label)
            pred2 = pred.cpu().detach().numpy()
            label2 = label.cpu().detach().numpy()
            acc = ssim(pred2[0, 0, :, :], label2[0, 0, :, :], data_range=1.0)
            acc1 = mse(pred2[0, 0, :, :], label2[0, 0, :, :])
            acc2 = psnr(pred2[0, 0, :, :], label2[0, 0, :, :], data_range=1.0)
            running_ssim.append(acc)
            running_psnr.append(acc2)
            running_loss.append(loss.item())
            # save_deartif_image(label.cpu().data,
            #                    f"./pred/label{epoch}.png")
            # save_deartif_image(image.cpu().data,
            #                    f"./pred/image{epoch}.png")
            # save_deartif_image(pred.cpu().data,
            #                     f"./pred/pred{epoch}.png")
        test_loss = np.sum(running_loss) / len(test_loader)
        test_mse = np.sum(running_mse) / (len(test_loader))
        test_ssim = np.sum(running_ssim) / (len(test_loader))
        test_psnr = np.sum(running_psnr) / (len(test_loader))
        std_psnr = np.std(running_psnr)
        std_ssim = np.std(running_ssim)
    print(f'test_psnr:{test_psnr}\ntest_ssim:{test_ssim}\nstd_psnr:{std_psnr}\nstd_ssim:{std_ssim}')
    # print('test_loss:',test_loss,'test_psnr:', test_psnr, 'test_ssim:', test_ssim)
    return test_loss,test_psnr,test_ssim,test_mse,running_loss,running_psnr,running_ssim,running_mse

def save_index(train_all_loss,train_loss,val_loss,val_psnr,val_ssim,test_loss,test_psnr,test_ssim,test_mse,running_loss,running_psnr,running_ssim,running_mse,index_path):
    torch.save({
                'train_all_loss':train_all_loss,
                'train_loss':train_loss,
                'val_loss':val_loss,
                'val_psnr':val_psnr,
                'val_ssim':val_ssim,
                'test_loss':test_loss,
                'test_psnr':test_psnr,
                'test_ssim':test_ssim,
                'test_mse':test_mse,
                'running_loss':running_loss,
                'running_psnr':running_psnr,
                'running_ssim':running_ssim,
                'running_mse':running_mse
                },index_path)

def kfold_train():
    for fold in range(5):
        print(fold)
        data_path = "./data/"
        save_path = "./model/model_pth/"  # 当前目录下#####################
        model_type = 0
        batch_size = 16
        learning_rate = 0.0001
        num_epochs = 500
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if model_type == 0:
            model = UNet(n_channels=1)
        elif model_type == 1:
            model = UNext(input_channels=1, img_size=224)
        else:
            args = get_config(config)
            # 明确指定要访问的属性名称
            # arch_name = 'SwinUNet'  # 模型名称
            # 获取模块中的属性
            # arch_class = getattr(archs, arch_name)
            # 实例化属性
            # model = arch_class(args)
            model = SwinUNet(config=args)
            model.load_from('./swin_tiny_patch4_window7_224.pth')
        model_name = type(model).__name__
        # 读取数据
        train_dataset = Train_load(os.path.join(data_path, f'fold_{fold}/train'))
        val_dataset = Train_load(os.path.join(data_path, f'fold_{fold}/val'))
        test_dataset = val_load(os.path.join(data_path, f'fold_{fold}/test'))
        train_loader = torch.utils.data.DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
        val_loader = torch.utils.data.DataLoader(batch_size=batch_size, dataset=val_dataset, shuffle=False)
        test_loader = torch.utils.data.DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=False)
        print('dataset prepared')
        # 模型训练设置
        model.to(device=device)
        optimizer = optim.Adam(model.parameters(), learning_rate)  # 使用L2范数,    # 优化算法, weight_decay=0.0001
        # optimizer = optim.AdamW(model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1,
                                                               verbose=True)
        criterion = nn.MSELoss()          # Loss算法
        best_loss = float('inf')  # best_loss统计，1qq2q初始化为正无穷
        all = 1
        early_stopping = EarlyStopping(save_path,fold,all)
        print('Training process is going to start')
        # 预训练
        train_loss = []
        val_loss = []
        val_psnr = []
        val_ssim = []
        # start = time.time()
        # for epoch in tqdm(range(num_epochs), total=num_epochs):
        #     train_epoch_loss = fit(model, device, train_loader, criterion, optimizer)
        #     train_loss.append(train_epoch_loss)
        #     print('train_epoch_loss:', train_epoch_loss)
        #     val_epoch_loss, val_epoch_psnr, val_epoch_ssim = validate(model, val_loader, criterion, device)
        #     val_loss.append(val_epoch_loss)
        #     val_psnr.append(val_epoch_psnr)
        #     val_ssim.append(val_epoch_ssim)
        #     print('val_epoch_loss:', val_epoch_loss, ' val_epoch_psnr:', val_epoch_psnr,' val_epoch_ssim:', val_epoch_ssim)
        #     early_stopping(val_epoch_loss, model,optimizer,epoch)  # 达到早停止条件时，early_stop会被置为True
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break  # 跳出迭代，结束训练
        #     scheduler.step(val_epoch_loss)
        #     print(optimizer.state_dict()['param_groups'][0]['lr'])
        # end = time.time()
        # print(f"Took {((end - start) / 60):.3f} minutes to train fold{fold}")
        #全部训练集训练
        print('****************************************************************')
        all = 0
        early_stopping = EarlyStopping(save_path, fold, all)
        model_path = os.path.join(save_path, f'{model_name}/{model_name}_fold{fold}_model.pth')
        checkpoint = torch.load(model_path,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1,
                                                               verbose=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        data_loader = torch.utils.data.DataLoader(batch_size=batch_size, dataset=dataset, shuffle=True)
        train_all_loss = []
        test_loss = []
        test_psnr = []
        test_ssim = []
        test_mse = []
        start = time.time()
        for epoch in tqdm(range(num_epochs), total=num_epochs):
            train_epoch_loss = fit(model, device, data_loader, criterion, optimizer)
            train_all_loss.append(train_epoch_loss)
            print('train_epoch_loss:', train_epoch_loss)
            test_epoch_loss,test_epoch_psnr,test_epoch_ssim,test_epoch_mse,running_loss,running_psnr,running_ssim,running_mse = test(model,criterion,test_loader,device)
            test_loss.append(test_epoch_loss)
            test_psnr.append(test_epoch_psnr)
            test_ssim.append(test_epoch_ssim)
            test_mse.append(test_epoch_mse)
            early_stopping(test_epoch_loss, model,optimizer,epoch)  # 达到早停止条件时，early_stop会被置为True
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break  # 跳出迭代，结束训练
            scheduler.step(test_epoch_loss)
            print(optimizer.state_dict()['param_groups'][0]['lr'])
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes to train_all fold{fold}")
        train_all_loss = []
        start = time.time()
        for epoch in tqdm(range(num_epochs), total=num_epochs):
            train_epoch_loss = fit(model, device, data_loader, criterion, optimizer)
            train_all_loss.append(train_epoch_loss)
            print('train_epoch_loss:', train_epoch_loss)
            # scheduler.step(train_epoch_loss)
            early_stopping(train_epoch_loss, model, optimizer, epoch)  # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
        end = time.time()
        # 测试
        print('****************************************************************')
        print('****************************************************************\n')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        test_loss0,test_psnr0,test_ssim0,test_mse0,running_loss,running_psnr,running_ssim,running_mse = test(model,criterion,test_loader,device)
        #保存数据
        # index_path = os.path.join(save_path,f'{model_name}/{model_name}_fold{fold}_index.pth')
        save_index(train_all_loss,train_loss,val_loss,val_psnr,val_ssim,
                   test_loss,test_psnr,test_ssim,test_mse,running_loss,running_psnr,running_ssim,running_mse,index_path)


if __name__ == "__main__":
    kfold_train()