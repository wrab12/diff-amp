import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
device = "cuda:0"
model_checkpoint1 = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint1)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1, num_labels=3000).cuda()#3000
        # for param in self.bert1.parameters():
        #     param.requires_grad = False
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(3000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)  # 0.3

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert1(input_ids=x['input_ids'].to(device),
                                     attention_mask=x['attention_mask'].to(device))
        # output_feature = bert_output["logits"]
        # print(output_feature.size())
        # output_feature = self.bn1(self.fc1(output_feature))
        # output_feature = self.bn2(self.fc1(output_feature))
        # output_feature = self.relu(self.bn3(self.fc3(output_feature)))
        # output_feature = self.dropout(self.output_layer(output_feature))
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        # return torch.sigmoid(output_feature),output_feature
        return torch.softmax(output_feature, dim=1)


def AMP(test_sequences, model):
    # 保持 AMP 函数不变，只处理传入的 test_sequences 数据
    max_len = 18
    test_data = tokenizer(test_sequences, max_length=max_len, padding="max_length", truncation=True,
                          return_tensors='pt')
    model = model.to(device)
    model.eval()
    out_probability = []
    with torch.no_grad():
        predict = model(test_data).cuda()
        out_probability.extend(np.max(np.array(predict.cpu()), axis=1).tolist())
        test_argmax = np.argmax(predict.cpu(), axis=1).tolist()
    id2str = {0: "non-AMP", 1: "AMP"}
    return id2str[test_argmax[0]], out_probability[0]


input_file = "seq.txt"
output_file = "seq_.txt"
pos_file = "pos.txt"

amp_count = 0
non_amp_count = 0


# 一次性读取整个文件
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# 加载模型
model = MyModel()
model.load_state_dict(torch.load("weight/best_model.pth"))
print('\nGeneration Start')
# 处理每一行数据
for line in tqdm(lines, total=len(lines), desc="Processing"):
    line = line.strip()
    result, probability = AMP(line, model)

    # 写入结果到输出文件
    with open(output_file, 'a') as outfile:
        outfile.write(f"{line} {result} {probability}\n")

    # 统计AMP和非AMP的数量
    if result == "AMP" and not any(char in line for char in ["0", "X", "Z", "x", "z"]):
        amp_count += 1
        with open(pos_file, 'a') as posfile:
            posfile.write(f"{line} {result} {probability}\n")
    else:
        non_amp_count += 1

print("\n AMP Generation Finished")
