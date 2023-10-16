import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import warnings
import tqdm
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
device = "cuda:0"
model_checkpoint1 = "facebook/esm2_t12_35M_UR50D"  # 初始



df_train1 = pd.read_csv('data/training_data.csv')
df_val = pd.read_csv('data/val_data.csv')

train_sequences1 = df_train1["Seq"].tolist()
train_labels1 = df_train1["Label"].tolist()
val_sequences = df_val["Seq"].tolist()
val_labels = df_val["Label"].tolist()


class MyDataset(Dataset):
    def __init__(self, dict_data) -> None:
        super(MyDataset, self).__init__()
        self.data = dict_data

    def __getitem__(self, index):
        return [self.data['text'][index], self.data['labels'][index]]

    def __len__(self):
        return len(self.data['text'])


train_dict1 = {"text": train_sequences1, 'labels': train_labels1}
val_dict = {"text": val_sequences, 'labels': val_labels}

epochs = 500
learning_rate = 0.0005
batch_size = 2048  # 1024

tokenizer1 = AutoTokenizer.from_pretrained(model_checkpoint1)  # model_checkpoint1 = "facebook/esm2_t12_35M_UR50D"#初始


def collate_fn(batch):
    max_len = 30  # 30
    pt_batch = tokenizer1([b[0] for b in batch], max_length=max_len, padding="max_length", truncation=True,
                          return_tensors='pt')

    labels = [b[1] for b in batch]
    return {'labels': labels, 'input_ids': pt_batch['input_ids'],
            'attention_mask': pt_batch['attention_mask']}


train_data1 = MyDataset(train_dict1)
val_data = MyDataset(val_dict)
train_dataloader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1, num_labels=3000)  # 3000
        for param in self.bert1.parameters():
            param.requires_grad = False
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(3000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)  # 0.3

    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert1(input_ids=x['input_ids'].to(device),
                                     attention_mask=x['attention_mask'].to(device))
        output_feature = self.dropout(bert_output["logits"])
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        return torch.softmax(output_feature, dim=1), output_feature


model = MyModel().cuda()
model = model.to(device)
# model.load_state_dict(torch.load("best_model.pth"))

# nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
train_epochs_acc = []
valid_epochs_acc = []

best_acc = 0
for epoch in range(epochs):
    model.train()
    train_epoch_loss = []
    tp1 = 0
    fn1 = 0
    tn1 = 0
    fp1 = 0
    for index, batch in enumerate(train_dataloader1):
        batchs = {k: v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs, _ = model(batchs)
        label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64),
                                            num_classes=2).float()  # 原始int64
        loss = criterion(outputs.to(device), label.to(device))

        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        for j in range(0, len(train_argmax)):
            if batchs["labels"][j] == 1:
                if batchs["labels"][j] == train_argmax[j]:
                    tp1 += 1
                else:
                    fn1 = fn1 + 1
            else:
                if batchs["labels"][j] == train_argmax[j]:
                    tn1 = tn1 + 1
                else:
                    fp1 = fp1 + 1

    train_acc = float(tp1 + tn1) / len(train_labels1)
    train_epochs_acc.append(train_acc)
    train_epochs_loss.append(np.average(train_epoch_loss))

    model.eval()
    valid_epoch_loss = []
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    Sensitivity = 0
    Specificity = 0
    MCC = 0
    AUC = 0
    true_labels = []
    features_list = []

    pred_prob = []
    with torch.no_grad():
        for index, batch in enumerate(val_dataloader):
            batchs = {k: v for k, v in batch.items()}
            outputs, output_feature = model(batchs)

            features_list.append(output_feature.cpu().numpy())

            label = torch.nn.functional.one_hot(torch.tensor(batchs["labels"]).to(torch.int64), num_classes=2).float()
            loss = criterion(outputs.to(device), label.to(device))
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            val_argmax = np.argmax(outputs.cpu(), axis=1)
            true_labels += batchs["labels"]  # 收集真实标签
            pred_prob += outputs[:, 1].tolist()
            # print("\n")
            # print(pred_prob)
            for j in range(0, len(val_argmax)):
                if batchs["labels"][j] == 1:
                    if batchs["labels"][j] == val_argmax[j]:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if batchs["labels"][j] == val_argmax[j]:
                        tn = tn + 1
                    else:
                        fp = fp + 1
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    auc_score = roc_auc_score(true_labels, pred_prob)
    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)
    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    val_acc = float(tp + tn) / len(val_labels)
    if val_acc >= best_acc:
        best_acc = val_acc
        print("best_acc is {}".format(best_acc))
        # torch.save(model.state_dict(), f"weight/best_model.pth")

    print(
        f'epoch:{epoch}, train_acc:{train_acc}, val_acc:{val_acc}, prec:{Precision} SE:{Sensitivity}, SP:{Specificity} ,f1:{F1} ,MCC:{MCC}, AUC:{auc_score}')
