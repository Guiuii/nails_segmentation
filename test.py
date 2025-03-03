import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import yaml
import os

def test(test_dataset, model):
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Выбираем случайные индексы для визуализации
    indices = random.sample(range(len(test_dataset)), config['visualization']['number'])

    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)  # Добавляем batch dimension
            output = model(image)

            # Применяем порог к предсказаниям
            output = (output > 0.5).float()

            # Переводим тензоры в numpy для визуализации
            image = image.squeeze().cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            label = label.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

    # Если output имеет 2 канала, выбираем только один (обычно класс 1)
            if output.ndim == 3:  # Проверяем, есть ли у тензора 3 измерения (C, H, W)
                output = output[1] 

            # Визуализация
            plt.figure(figsize=(15, 5))

            # Оригинальное изображение
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.axis('off')
            plt.imshow(image)

            # Маска (лейбл)
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.axis('off')
            plt.imshow(label, cmap='gray')

            # Предсказание модели
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.axis('off')
            plt.imshow(output, cmap='gray')

            save_path = os.path.join(
            config['visualization']['results_dir'],
            f'segmentation_pic_{idx}.png')

            plt.savefig(save_path, bbox_inches="tight")