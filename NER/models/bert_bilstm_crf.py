import torch
import torch.nn as nn  # 搭建神经网络
from transformers import BertModel  # BERT模型
from torchcrf import CRF  # CRF层


class BertBiLSTMCRF(nn.Module):

    def __init__(self,  label_num, vocab_size, embedding_dim=768, hidden_dim=256):
        super(BertBiLSTMCRF, self).__init__()
        self.label_num = label_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bert = BertModel.from_pretrained('./bert/bert_base')
        self.bert_projection = nn.Linear(embedding_dim, hidden_dim)  # 添加的投影层
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)

        self.crf = CRF(self.label_num, batch_first=True)  # CRF层
        self.hidden2tag = nn.Linear(hidden_dim, self.label_num)


    def forward(self, data,is_train=True):
        device = next(self.bert.parameters()).device
        attention_mask = data['attention_mask'].to(device)
        input_ids = data['bert_ids'].to(device)
        labels = data['labels'].to(device)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state
        projected_embeddings = self.bert_projection(embeddings)  # 投影到与LSTM期望的输入大小相同的维度
        lstm_out, _ = self.lstm(projected_embeddings)
        emissions = self.hidden2tag(lstm_out)

        if is_train:
            loss = -self.crf(emissions, labels, reduction='mean', mask=attention_mask.bool().to(device))
            return loss
        else:
            return self.crf.decode(emissions,mask=attention_mask.bool().to(device))
