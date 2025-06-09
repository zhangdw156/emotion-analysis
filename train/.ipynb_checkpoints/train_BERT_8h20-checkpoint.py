# -*- coding: utf-8 -*-

import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')  # 无界面后端，适合服务器


# 参数配置
class Config:
    pretrained_model_path = "bert-base-uncased"  # 自动下载，无需本地路径
    train_data_path = "./data/train.json"
    val_data_path = "./data/validation.json"
    per_gpu_batch_size = 80  # 单卡批量大小（8卡总批量=80*8=640）
    max_length = 128
    num_epochs = 100
    learning_rate = 2e-5
    num_labels = 6  # 6种情感类别
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()  # 自动获取GPU数量
    distributed = n_gpus > 1  # 是否启用多卡训练


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
        text = "Please identify the emotions contained in the following text and output the corresponding labels. Label 0 corresponds to sadness, label 1 corresponds to joy, label 2 corresponds to love, label 3 corresponds to anger, label 4 corresponds to fear, label 5 corresponds to surprise." + " " + item['text']
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    config = Config()
    
    # 初始化分布式环境（仅主卡执行后续代码）
    if config.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        config.device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0

    # 加载模型和分词器（自动下载到缓存目录）
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_path)
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=config.num_labels
    ).to(config.device)

    # 多卡包裹模型（仅非主卡不参与保存/日志）
    if config.distributed:
        model = DDP(model, device_ids=[config.device])

    # 数据集和数据加载器（分布式采样）
    train_dataset = EmotionDataset(config.train_data_path, tokenizer, config.max_length)
    val_dataset = EmotionDataset(config.val_data_path, tokenizer, config.max_length)

    train_sampler = DistributedSampler(train_dataset) if config.distributed else None
    val_sampler = DistributedSampler(val_dataset) if config.distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_gpu_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,  # 建议设置数据加载线程数
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # 训练循环（仅主卡记录日志和保存模型）
    if local_rank == 0:
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_f1s = []
        val_f1s = []

    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)  # 确保分布式采样的随机性

        # 训练和评估
        train_loss, train_acc, train_f1 = train(model, train_dataloader, optimizer, scheduler, config.device)
        val_loss, val_acc, val_f1 = evaluate(model, val_dataloader, config.device)

        # 仅主卡处理日志和保存
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
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
                model.module.save_pretrained("./best_model_BERT")  # 保存DDP包裹的模型参数
                tokenizer.save_pretrained("./best_model_BERT")
                print("Best model saved!")

    # 仅主卡保存最终结果和图表
    if local_rank == 0:
        datetime_str = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        dir_name = f'./history_model/emotion_calculate_BERT_{datetime_str}'
        visual_dir = f"./history_model/visualization_BERT_{datetime_str}"
        os.makedirs(visual_dir, exist_ok=True)
        model.module.save_pretrained(dir_name)  # 保存DDP模型的原始参数
        tokenizer.save_pretrained(dir_name)

        plot_result(train_losses, val_losses, visual_dir)
        plot_acc(train_accs, val_accs, visual_dir)
        plot_f1(train_f1s, val_f1s, visual_dir)
        print("\nTraining completed!")

    # 清理分布式环境
    if config.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()