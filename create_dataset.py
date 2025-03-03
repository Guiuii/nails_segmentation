from torch.utils.data import Dataset
from PIL import Image
import os

class NailsDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_extensions=['.jpg', '.jpeg', '.png']):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_extensions = image_extensions

        # Собираем только файлы с указанными расширениями
        self.image_names = [
            fname for fname in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, fname)) and fname.lower().endswith(tuple(image_extensions))
        ] # список названий всех картинок с удовлетворяющим расширением

    def __len__(self):
        return len(self.image_names) # возвращает количество картинок

    def __getitem__(self, idx): # обращение к изображению и его метке по индексу
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        # Загружаем изображение и маску
        image = Image.open(img_path).convert("RGB")  # Убедимся, что изображение в RGB
        label = Image.open(label_path).convert("L")  # Убедимся, что маска в градациях серого

        if self.transform:
            image = self.transform(image)
            
            # Применяем трансформации к маске, но не нормализуем
            label = self.transform(label)

        # Преобразуем маску в бинарную (0 или 1)
        label = (label > 0.5).float()  # Порог 127 для значений [0, 255]
        
        # Бинарная маска — это изображение, где каждый пиксель принадлежит к одному из двух классов:
        # 0: фон
        # 1: ногти

        return image, label