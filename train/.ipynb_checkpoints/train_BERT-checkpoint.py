# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')


# 参数配置
class Config:
    pretrained_model_path = "./model/bert-base-uncased"
    train_data_path = "./data/train.json"
    val_data_path = "./data/validation.json"
    batch_size = 256  # 根据8GB显存调整
    max_length = 128
    num_epochs = 100
    learning_rate = 2e-5
    num_labels = 6  # 6种情感类别
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 合并instruction和input作为模型输入
        # text = item['instruction'] + " " + item['input']
        text = "Please identify the emotions contained in the following text and output the corresponding labels. Label 0 corresponds to sadness, label 1 corresponds to joy, label 2 corresponds to love, label 3 corresponds to anger, label 4 corresponds to fear, label 5 corresponds to surprise." + " " + \
               item['text']
        label = int(item['label'])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 训练函数
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        logits = outputs.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    return avg_loss, accuracy, f1


# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    return avg_loss, accuracy, f1


def plot_result(train_losses, val_losses, dir_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_name}/loss.png")


def plot_acc(train_accs, val_accs, dir_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_name}/accuracy.png")

def plot_f1(train_f1s, val_f1s, dir_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Training and Validation F1 Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir_name}/f1_score.png")


def main():
    # 初始化配置
    config = Config()

    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_path)
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=config.num_labels
    ).to(config.device)

    # 加载数据集
    train_dataset = EmotionDataset(config.train_data_path, tokenizer, config.max_length)
    val_dataset = EmotionDataset(config.val_data_path, tokenizer, config.max_length)

    # 划分训练集和验证集 (80-20分割)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # 训练
        train_loss, train_acc, train_f1 = train(model, train_dataloader, optimizer, scheduler, config.device)

        # 验证
        val_loss, val_acc, val_f1 = evaluate(model, val_dataloader, config.device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("./best_model_BERT")
            tokenizer.save_pretrained("./best_model_BERT")
            print("Best model saved!")

    datetime_str = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    dir_name: str = f'./history_model/emotion_calculate_BERT_{datetime_str}'
    visual_dir: str = f"./history_model/visualization_BERT_{datetime_str}"
    os.mkdir(visual_dir)
    model.save_pretrained(dir_name)
    tokenizer.save_pretrained(dir_name)

    plot_result(train_losses, val_losses, visual_dir)
    plot_acc(train_accs, val_accs, visual_dir)
    plot_f1(train_f1s, val_f1s, visual_dir)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
