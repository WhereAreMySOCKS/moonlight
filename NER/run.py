import argparse
import logging
import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
import warnings

from tqdm import tqdm

from NER.models.bert_bilstm_crf import BertBiLSTMCRF
from NER.models.bert_cnn_bilstm_crf import BertCNNBiLSTMCRF
from NER.models.cnn_bilstm_crf import CNNBiLSTMCRF
from NER.models.fusion_model import FusionModel
from utils import configure_logging

warnings.filterwarnings("ignore") # 忽略警告信息


def train_model(model, optimizer, train_loader, val_loader, args):
    logging.info(f" ==== Start training {type(model).__name__} model ====")
    best_val_acc = 0
    for e in range(args.epoch):
        model.train()
        running_loss = []
        for data in train_loader:
            loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item()/len(data))

        epoch_train_loss = np.mean(running_loss)
        # 验证模型并计算验证损失
        current_loss = evaluate(model, args.device, val_loader)
        logging.info(f"Epoch {e + 1}/{args.epoch}, Train Loss: {epoch_train_loss}, Val Loss: {current_loss}")

        # 保存最佳模型
        if current_loss['acc'] > best_val_acc:
            best_val_acc = current_loss['acc']
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path) # 创建保存模型权重的文件夹
            torch.save(model.state_dict(), args.save_path+f'{type(model).__name__}.pth') # 更新最佳模型权重

@ torch.no_grad()
def evaluate(_model, _device, _val_loader):
    _model.eval()
    labels = []
    prediction = []
    for data in _val_loader:
        mask = data['attention_mask'].numpy()
        pre = _model(data,is_train=False)
        label = data['labels'].numpy()
        # 用mask过滤pre和label
        selected_label= label[mask != 0]
        prediction.extend([item for sublist in pre for item in sublist])
        labels.extend(selected_label.flatten())
    print('预测结果中标签数量: ',len(set(prediction)))
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction, average='macro')
    metrics = {'acc': acc, 'f1': f1}
    return metrics

def train(args):
    # 加载词表
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    # 加载标签
    label2id = pickle.load(open(args.label2id_path, 'rb'))

    train_dataset = pickle.load(open(args.train_data_path, 'rb')) # 读取先前保存的训练集
    train_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=train_dataset, shuffle=True)
    val_dataset = pickle.load(open(args.val_data_path, 'rb')) # 读取先前保存的验证集
    val_loader = torch.utils.data.DataLoader(batch_size=args.batch_size, dataset=val_dataset, shuffle=False)
    model = None
    if args.model_type == 'bert_bilstm_crf':
        model = BertBiLSTMCRF(label_num =len(label2id),
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'cnn_bilstm_crf':
        model = CNNBiLSTMCRF(label_num =len(label2id),
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'bert_cnn_bilstm_crf':
        model = BertCNNBiLSTMCRF(label_num =len(label2id),
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'fusion_model':
        model = FusionModel(label_num =len(label2id),
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim).to(args.device)
    # record the number of parameters
    logging.info(f"Model {type(model).__name__} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    # 分离 BERT 层和其他层的参数
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    # 定义优化器，使用参数组来设置不同的学习率
    optimizer = optim.Adam([
        {'params': bert_params, 'lr': args.learning_rate},
        {'params': other_params, 'lr': args.learning_rate * 5}
    ])

    train_model(model, optimizer, train_loader, val_loader, args)

@torch.no_grad()
def predict(args):
    # 加载词表
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    # 反转k,v
    label2id = pickle.load(open(args.label2id_path, 'rb'))
    id2label = {v: k for k, v in label2id.items()}


    test_dataset = pickle.load(open(args.test_data_path, 'rb'))  # 读取先前保存的训练集
    test_loader = torch.utils.data.DataLoader(batch_size=args.batch_size * 30, dataset=test_dataset, shuffle=False)
    model = None
    if args.model_type == 'bert_bilstm_crf':
        model = BertBiLSTMCRF(label_num=len(label2id),
                              vocab_size=len(vocab),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'cnn_bilstm_crf':
        model = CNNBiLSTMCRF(label_num=len(label2id),
                             vocab_size=len(vocab),
                             embedding_dim=args.embedding_dim,
                             hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'bert_cnn_bilstm_crf':
        model = BertCNNBiLSTMCRF(label_num=len(label2id),
                                 vocab_size=len(vocab),
                                 embedding_dim=args.embedding_dim,
                                 hidden_dim=args.hidden_dim).to(args.device)
    elif args.model_type == 'fusion_model':
        model = FusionModel(label_num=len(label2id),
                            vocab_size=len(vocab),
                            embedding_dim=args.embedding_dim,
                            hidden_dim=args.hidden_dim).to(args.device)

    model.load_state_dict(torch.load(args.save_path+f'{type(model).__name__}.pth'))
    model.eval()

    with open(f'./data/dataset/{type(model).__name__}_predict.txt', 'w') as f:
        for data in test_loader:
            pre = model(data, is_train=False)
            for text,item in zip(data['text'],pre):
                for char,l in zip(list(text),item[1:]):
                    # 写入预测结果
                        f.write(f'{char} {id2label[l]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--log_path', type=str, default="./log/", help='path to save log.')
    parser.add_argument('--train_data_path', type=str, default="./data/dataset/train.pkl", help='path to train dataset.')
    parser.add_argument('--test_data_path', type=str, default="./data/dataset/test.pkl", help='path to test dataset.')
    parser.add_argument('--val_data_path', type=str, default="./data/dataset/val.pkl", help='path to val dataset.')
    parser.add_argument('--vocab_path', type=str, default="./data/dataset/vocab.pkl", help='path to vocab.')
    parser.add_argument('--label2id_path', type=str, default="./data/dataset/label2id.pkl", help='path to label2id.')
    parser.add_argument('--embedding_dim', type=int, default=768, help='embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension.')
    parser.add_argument('--save_path', type=str, default='./save_model/', help='path to save model.')

    parser.add_argument('--gpu', type=str, default='7', help='gpu device index.')
    parser.add_argument('--epoch', type=int, default=30, help='number of epoc  hs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate.')
    parser.add_argument('--model_type', type=str, default='cnn_bilstm_crf', help='Please choose from [cnn_bilstm_crf, '
                                                                                 'bert_bilstm_crf,bert_cnn_bilstm_crf,fusion_model]')
    arg = parser.parse_args()
    arg.device = torch.device('cuda:'+arg.gpu) if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    # 构建日志文件的路径
    log_filename = f"{arg.log_path}{arg.model_type}.log"
    configure_logging(log_filename,mod='w')
    # 记录参数信息
    logging.info(f" ==== Using {arg.model_type} model ==== ")
    logging.info(
        f'batch_size: {arg.batch_size}, lr for bert: {arg.learning_rate}, lr for other layer: {arg.learning_rate * 5}, epoch: {arg.epoch}')

    train(arg) # 训练模型
    predict(arg) # 评估模型
