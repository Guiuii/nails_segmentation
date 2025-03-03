import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import kagglehub
import yaml
from create_dataset import NailsDataset
from train import train
from test import test
from plot_metrics import plot_metrics

# Загрузка конфига
with open("config.yaml") as f:
    config = yaml.safe_load(f)
    
# Загрузка последней версии
path = kagglehub.dataset_download(
    config["data"]["dataset_path"])

# Определение траснформаций 
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor()          
])

# Подготовка датасета
dataset = NailsDataset(image_dir=os.path.join(path, 'images'), label_dir=os.path.join(path, 'labels'), transform=transform)

train_size = int(config['data']['train_ratio'] * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

# Обучение модели
metrics, model = train(train_loader, dataset)

# Построение графиков изменения метрик в зависимости от количества эпох 
plot_metrics(metrics)

# Тестирование модели
test(test_dataset, model)