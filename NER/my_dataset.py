"""
NER任务的dataset类
"""
import torch
from torch.utils.data import  Dataset
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, texts,text_ids, labels, max_len):
        self.tokenizer = BertTokenizer.from_pretrained('./bert/bert_base')
        self.texts = texts
        # list(list) -> torch.tensor(torch.tensor)
        self.texts_ids = [torch.tensor(i) for i in text_ids]
        self.labels = labels
        self.max_len = max_len

        # 对所有文本进行编码
        self.tokenizer_output = self.tokenizer(
            texts, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True,
            return_tensors='pt'
        )
        self.input_ids = self.tokenizer_output['input_ids']
        self.attention_mask = self.tokenizer_output['attention_mask']
        # 将标签转换为tensor
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        # texts和labels的长度是一致的
        return len(self.texts)

    def __getitem__(self, index):
        # 根据索引返回一个样本
        return {
            'bert_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'one_hot': self.texts_ids[index],
            'labels': self.labels[index],
            'text': self.texts[index]
        }
