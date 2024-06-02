import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
from transformers import BertModel


class FusionModel(nn.Module):
    def __init__(self, label_num, vocab_size, embedding_dim=300, hidden_dim=256, kernel_sizes=[3]):
        super( FusionModel, self).__init__()
        self.label_num = label_num
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bert = BertModel.from_pretrained('./bert/bert_base')
        self.conv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=[3,embedding_dim],padding=[1,0])
        self.bert_projection = nn.Linear(embedding_dim, hidden_dim)  # 添加的投影层

        self.bilstm = nn.LSTM(hidden_dim * len(kernel_sizes), hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_num)
        self.crf = CRF(num_tags=label_num, batch_first=True)

    def forward(self, data,is_train=True):
        device = next(self.bilstm.parameters()).device
        labels = data['labels'].to(device)
        input_ids = data['bert_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)

        embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # you can also use nn.Embedding to get embedding and then use conv to get cnn_outputs
        cnn_outputs = self.conv(embedding.unsqueeze(1)).squeeze(3).permute(0,2,1)
        bert_embedding = self.bert_projection(embedding)
        # concat or add
        fusion_embedding = cnn_outputs + bert_embedding
        lstm_inputs = self.dropout(fusion_embedding)
        lstm_outputs, _ = self.bilstm(lstm_inputs)

        emissions = self.hidden2tag(lstm_outputs.squeeze(1))

        if is_train:
            loss = -self.crf(emissions, labels, mask=data['attention_mask'].bool().to(device), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions,mask=data['attention_mask'].bool().to(device))



