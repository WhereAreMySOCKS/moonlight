"""
这部分的训练逻辑有问题 训练后模型参数发生变化，command中文本的表示已经不能用了
效果很差 直接相似度还比较准。。。
"""
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class SentenceSimilarityModel(torch.nn.Module):
      def __init__(self, device, command_path):
            super(SentenceSimilarityModel, self).__init__()
            self.device = device
            self.command_path = command_path
            self.text_command_pair = None
            self.text_embeddings = None
            self.bert = BertModel.from_pretrained('./text_match/model/bert_base').to(self.device)
            self.mlp = nn.Sequential(
                  nn.Linear(768 * 2, 256),
                  nn.ReLU(),
                  nn.Linear(256, 1),
                  nn.Sigmoid()
            )
            self.tokenizer = BertTokenizer.from_pretrained('./text_match/model/bert_base')
            self.get_target_embeddings()
            self.to(self.device)

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
      def match(self, text):
            token = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            encoded, mask = token['input_ids'].to(self.device), token['attention_mask'].to(self.device)
            output = self.bert(encoded, mask)['pooler_output']
            probs = []
            for i in self.text_embeddings:
                  similarity_score = self.forward(output, i.unsqueeze(0)).cpu().item()  # 获取相似度分数并转移到CPU
                  probs.append(similarity_score)
                  # 返回概率最大的索引
            most_similar_index = probs.index(max(probs))
            most_similar_command_k = list(self.text_command_pair.keys())[most_similar_index]
            most_similar_command_value = self.text_command_pair[most_similar_command_k]
            return most_similar_command_k, most_similar_command_value

      @torch.no_grad()
      def calculate_similarity(self, text_1, text_2):
            text_1_output, text_2_output = self.compute_embedding(text_1, text_2)
            return torch.nn.functional.cosine_similarity(text_1_output, text_2_output)
