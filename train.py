import torch
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from compute_class_weights import compute_class_weights
from unet import UNet

def train(train_loader, dataset):

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # Вычисление весов классов (для учета несбалансированности)
    class_weights = compute_class_weights(dataset).to(device)
    print(f"Class Weights: {class_weights} [background, nails] \n")
    
    # Передаём веса в функцию потерь
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Передача весов в функцию потерь нужна для балансировки вклада каждого класса в обучение модели
    # Искуственно завышаем ошибку для представителей меньшего по объему класса

    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr']) 

    full_loss = []
    full_f1 = []

    num_epochs = config["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train() # Переводим модель в режим обучения
        epoch_loss = 0
        epoch_f1 = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Обнуление градиентов параметров модели перед каждым шагом обучения
            outputs = model(images) # (2, H, W)
            # Каждый пиксель описывается двумя значениями (принадлежность к классу фон (0) или ноготь (1))

            labels = (labels > 0.5).long()  
            if len(labels.shape) == 4:  
                labels = labels.squeeze(1)
            # squeeze(1) удаляет лишнюю размерность из тензора масок (размерность по индексу 1, если она есть)
            # labels.shape = (Batch, 1, Height, Width), после squeeze(1) получим (Batch, Height, Width).

            # Вычисление потерь
            #loss = criterion(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward() # Вычисление градиентов функции потерь по параметрам модели
            optimizer.step() # Обновление весов на основе вычисленных градиентов

            epoch_loss += loss.item() # Значение loss для batch

            # Вычисление F1 score для оценки качества модели
            predicted = torch.argmax(outputs, dim=1)  # Применяем argmax для многоклассовой сегментации
            # Для каждого пикселя выбирает класс с максимальной вероятностью (0 или 1)

            f1 = f1_score(labels.cpu().numpy().flatten(), predicted.cpu().numpy().flatten(), average='binary')
            # .flatten(): преобразует многомерный массив NumPy в одномерный
            epoch_f1 += f1

        full_loss.append(epoch_loss / len(train_loader))
        full_f1.append(epoch_f1 / len(train_loader))


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, F1 Score: {epoch_f1 / len(train_loader):.4f}")
        
    metrics =  {'Loss': full_loss,'F1-score': full_f1}
        
    return metrics, model