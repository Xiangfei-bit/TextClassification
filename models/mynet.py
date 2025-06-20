# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'mynet'
        self.train_path = dataset + './data/train.txt'  # 训练集
        self.dev_path = dataset + './data/dev.txt'  # 验证集
        self.test_path = dataset + './data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + './data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + './data/vocab.pkl'  # 词表
        self.save_path = dataset + './saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + './log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + './data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.hidden_size2 = 64

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 嵌入层
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        # 第一层双向LSTM
        self.lstm1 = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.attention1 = nn.Linear(config.hidden_size * 2, 1)  # 注意力权重计算

        # 输入重加权层 (对应论文中的Input Re-weight模块)
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.Sigmoid()
        )

        # 第二层双向LSTM
        self.lstm2 = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh2 = nn.Tanh()
        self.attention2 = nn.Linear(config.hidden_size * 2, 1)  # 第二层注意力权重计算

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),  # 拼接两层特征
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size2, config.num_classes)
        )

    def forward(self, x):
        x, _ = x
        # 嵌入层处理
        emb = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # 第一层LSTM和注意力
        H1, _ = self.lstm1(emb)  # [batch_size, seq_len, hidden_size*2]
        M1 = self.tanh1(H1)  # 非线性变换
        alpha1 = F.softmax(self.attention1(M1), dim=1)  # 注意力权重 [batch_size, seq_len, 1]

        # 输入重加权 (核心改进点)
        gate_weights = self.gate(H1)  # 计算门控权重
        reweighted_input = H1 * gate_weights * alpha1  # 应用门控和注意力权重

        # 第二层LSTM和注意力
        H2, _ = self.lstm2(reweighted_input)  # 处理重加权后的输入
        M2 = self.tanh2(H2)
        alpha2 = F.softmax(self.attention2(M2), dim=1)  # 第二层注意力权重

        # 特征融合 (结合两层注意力的信息)
        # 1. 全局特征: 对序列维度求和并池化
        global_feature1 = torch.sum(H1 * alpha1, dim=1)  # [batch_size, hidden_size*2]
        global_feature2 = torch.sum(H2 * alpha2, dim=1)  # [batch_size, hidden_size*2]

        # 2. 拼接两层特征
        combined_feature = torch.cat([global_feature1, global_feature2], dim=1)  # [batch_size, hidden_size*4]

        # 分类输出
        logits = self.fc(combined_feature)

        return logits, (alpha1, alpha2)  # 返回分类结果和两层注意力权重


