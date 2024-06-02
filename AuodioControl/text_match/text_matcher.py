import torch
from transformers import BertTokenizer, BertModel
import json

class TextMatcher:
    def __init__(self, device,model_path,command_path):
        self.text_command_pair = None
        self.text_embeddings = None
        self.device = device
        self.command_path = command_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path).to(self.device)
        self.get_target_embeddings()

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

    @torch.no_grad()
    def match(self, pattern):
        # 将输入文本转换为模型的输入特征
        inputs = self.tokenizer(pattern, return_tensors='pt').to(self.device)
        # 使用模型获取输入文本的特征表示
        with torch.no_grad():
            outputs = self.bert(**inputs)
            input_embedding = outputs.last_hidden_state.mean(dim=1)  # 取最后一个隐藏层的均值作为嵌入表示

        # 计算余弦相似度
        similarity = torch.cosine_similarity(input_embedding, self.text_embeddings, dim=1)
        # 获取最相似的命令的索引
        max_similarity_index = similarity.argmax().item()
        # 返回最相似的命令的值
        most_similar_command_k = list(self.text_command_pair.keys())[max_similarity_index]
        most_similar_command_value = self.text_command_pair[most_similar_command_k]
        return most_similar_command_k,most_similar_command_value

