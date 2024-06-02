import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer,BertModel
import pickle as pkl

import json

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class SentenceSimilarityModel(torch.nn.Module):
      def __init__(self, device,command_path):
            super(SentenceSimilarityModel, self).__init__()
            self.device = device
            self.command_path = command_path
            self.text_command_pair = None
            self.text_embeddings = None
            self.bert = BertModel.from_pretrained('../text_match/model/simbert-base').to(self.device)
            self.mlp = nn.Sequential(
                  nn.Linear(768 * 2, 256),
                  nn.ReLU(),
                  nn.Linear(256, 1),
                  nn.Sigmoid()
            )
            self.tokenizer = BertTokenizer.from_pretrained('../text_match/model/simbert-base')


      @torch.no_grad()
      def get_target_embeddings(self):
            with open(self.command_path, 'r', encoding='utf-8') as f:
                  self.text_command_pair = json.load(f)
            # 将所有目标命令的文本转换为模型的输入特征，并获取嵌入表示
            text_inputs = self.tokenizer(list(self.text_command_pair.keys()), return_tensors='pt', padding=True,
                                         truncation=True).to(self.device)
            target_outputs = self.bert(**text_inputs)
            target_embeddings = target_outputs.last_hidden_state.mean(dim=1)  # 取最后一个隐藏层的均值作为嵌入表示
            self.text_embeddings = target_embeddings  # 存储嵌入表示

      def compute_embedding(self, text_1, text_2):
            # 编码文本和正负例句子
            token_1 = self.tokenizer(text_1, return_tensors='pt', padding=True, truncation=True)
            token_2 = self.tokenizer(text_2, return_tensors='pt', padding=True, truncation=True)

            text_1_encoded, t1_mask = token_1['input_ids'].to(self.device), token_1['attention_mask'].to(self.device)
            text_2_encoded, t2_mask = token_2['input_ids'].to(self.device), token_2['attention_mask'].to(self.device)

            # 获取BERT输出
            text_1_output = self.bert(text_1_encoded, t1_mask)['pooler_output']
            text_2_output = self.bert(text_2_encoded, t2_mask)['pooler_output']
            return text_1_output, text_2_output

      def forward(self, text_1, text_2):
          if type(text_1) != torch.Tensor:
            text_1_output, text_2_output = self.compute_embedding(text_1, text_2)
            mlp_input = torch.cat([text_1_output, text_2_output], dim=1)
          else:
              mlp_input = torch.cat([text_1, text_2], dim=1)
          return self.mlp(mlp_input)

      @torch.no_grad()
      def match(self,text):
          self.get_target_embeddings()
          token = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
          encoded, mask = token['input_ids'].to(self.device), token['attention_mask'].to(self.device)
          output = self.bert(encoded, mask)['pooler_output']
          probs = []
          for i in self.text_embeddings:
              similarity_score = self.forward(output, i.unsqueeze(0)).cpu().item()  # 获取相似度分数并转移到CPU
              probs.append(similarity_score)
              most_similar_index = probs.index(max(probs)) # 返回概率最大的索引
              most_similar_command_k = list(self.text_command_pair.keys())[most_similar_index]
              most_similar_command_value = self.text_command_pair[most_similar_command_k]
              return most_similar_command_k, most_similar_command_value



      @torch.no_grad()
      def calculate_similarity(self, text_1, text_2):
            text_1_output, text_2_output = self.compute_embedding(text_1, text_2)
            return torch.nn.functional.cosine_similarity(text_1_output, text_2_output)

# 定义数据集类
class SentenceSimilarityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def split_train_test_val(data, val_split=0.1,test_split=0.1):
    # 划分训练集，测试集合，验证集
    random.shuffle(data)
    val_size = int(len(data) * val_split)
    test_size = int(len(data) * test_split)
    val_data = data[:val_size]
    test_data = data[val_size:val_size + test_size]
    train_data = data[val_size + test_size:]
    return train_data, val_data, test_data



# 定义训练函数
def train_model(_device,_model, all_data, batch_size=32, num_epochs=5, learning_rate=1e-4):
    _model.to(_device)
    _model.train()
    # 冻结bert参数
    # for param in _model.bert.parameters():
    #     param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss() # 损失函数

    train_dataset = SentenceSimilarityDataset(all_data[0])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SentenceSimilarityDataset(all_data[1])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    best_loss = 10000
    for epoch in range(num_epochs):
        total_loss = 0.0
        step = 0
        for text1, text2, label in train_dataloader:
            step += 1
            label = label.to(device)
            optimizer.zero_grad()
            pre = model(text1, text2).squeeze()
            loss = criterion(pre, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if epoch > 1 and step % 20 == 0:
                eval_loss = evaluate_model(device , model, val_dataloader)
                print(f"Epoch {epoch+1}, Step {step}, Loss: {total_loss / 100}, Eval Loss: {eval_loss}")
                if best_loss > eval_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), './data/model.pth')
                    print("model is saved!")

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")


# 定义验证函数
@torch.no_grad()
def evaluate_model(_device,_model, dataloader):
    _model.to(_device)
    _model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    with torch.no_grad():
        for text1, text2, label in dataloader:
            label = label.to(_device)
            pre = _model(text1, text2).squeeze()
            loss = criterion(pre, label.float())
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader)}")
    return  total_loss

@torch.no_grad()
def metrics(test_device, test_data,batch_size=32):
    # 加载模型
    model_trained = SentenceSimilarityModel(test_device,'c.json').to(test_device)
    model_trained.load_state_dict(torch.load('./data/model.pth'))
    model_trained.eval()
    test_dataset = SentenceSimilarityDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    label_list,pre_list, pre_cos_list= [],[],[]
    for text1, text2, label in test_dataloader:
        pre = model_trained(text1, text2).squeeze().cpu()
        # 根据概率选择标签为0还是1
        pre_list.append(torch.where(pre > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        cos_sim = model_trained.calculate_similarity(text1, text2).cpu()
        pre_cos_list.append(torch.where(cos_sim > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        label_list.append(label)

    label_list = torch.cat(label_list)
    pre_list = torch.cat(pre_list)
    pre_cos_list = torch.cat(pre_cos_list)
    # 计算acc
    acc_1 = torch.sum(pre_list == label_list).item() / len(label_list)
    acc_2 = torch.sum(pre_cos_list == label_list).item() / len(label_list)
    print(f"Trained Model Accuracy: {acc_1}")
    print(f"Trained Model Cosine Similarity Accuracy: {acc_2}")
    return model_trained


if __name__ == '__main__':
    # 打开data.txt，读取每行字符串，将其分割为三部分，分别为命令、正例、负例，然后将其保存到data列表中。
    all_data = split_train_test_val(pkl.load(open('./data/result.pkl', 'rb')))
    print("训练，测试，验证集大小分别为: ",len(all_data[0]),len(all_data[1]),len(all_data[2]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SentenceSimilarityModel(device,'c.json').to(device)
    model.load_state_dict(torch.load('./data/model.pth'))
    train_model(device,model, all_data, batch_size=64, num_epochs=5, learning_rate=1e-5)
    model = metrics(device,all_data[2])


