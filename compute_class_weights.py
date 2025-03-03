import numpy as np
import torch

def compute_class_weights(dataset):
    ''' Функция для подсчёта весов классов '''
    count_0, count_1 = 0, 0

    for img, mask in dataset:
        mask_np = mask.numpy()  # Преобразуем маску в numpy array
        count_0 += np.sum(mask_np == 0) # кол-во пикселей, принадлежащих фону
        count_1 += np.sum(mask_np == 1) # кол-во пикселей, принадлежащих ногтям

    total_pixels = count_0 + count_1
    freq_0 = count_0 / total_pixels # частота встречаемости пикселей фона
    freq_1 = count_1 / total_pixels # частота встречаемости пикселей ногтей

    weight_0 = 1 / freq_0 # вес класса фона
    weight_1 = 1 / freq_1 # вес класса ногтей

    return torch.tensor([weight_0, weight_1], dtype=torch.float32)