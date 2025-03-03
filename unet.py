import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential( # контейнер, который позволяет объединять несколько операций в одну структуру
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #изображение не уменьшаеься за счет паддинга
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Блоки сверток, последовательно увеличивающие количество каналов
        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        # После каждого блока сверток применяется операция макс-пулинга, которая уменьшает размер изображения в 2 раза.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок, который обрабатывает самые глубокие и абстрактные признаки
        # Размеры минимальны, количество каналов максимально
        self.bottleneck = conv_block(512, 1024)

        # Транспонированная свертка увеличивает размер изображения в 2 раза
        self.upconv4 = up_conv(1024, 512)
        # Блоки сверток, которые обрабатывают объединенные тензоры и уменьшают количество каналов
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)

        # Изменен выход, теперь два канала
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)  # 2 канала для фона и объекта

    def forward(self, x):
        # x=(3, 256, 256)
        
        # энкодер
        enc1 = self.encoder1(x) # (64, 256, 256)
        enc2 = self.encoder2(self.pool(enc1)) # (64, 128, 128) -> (128, 128, 128) 
        enc3 = self.encoder3(self.pool(enc2)) # (128, 64, 64) -> (256, 64, 64)
        enc4 = self.encoder4(self.pool(enc3)) # (256, 32, 32) -> (512, 32, 32)

        # бутылочное горлышко
        bottleneck = self.bottleneck(self.pool(enc4)) # (512, 16, 16) -> (1024, 16, 16)

        # декодер
        dec4 = self.upconv4(bottleneck) # (512, 32, 32)
        # torch.cat - skip-connections с соответствующим тензором из кодировщика 
        # Они помогают сети сохранять детали изображения, которые могли быть потеряны в кодировщике
        # Конкатенация по каналам
        dec4 = torch.cat((dec4, enc4), dim=1) # (1024, 32, 32) = (512, 32, 32) + (512, 32, 32)
        dec4 = self.decoder4(dec4) # (512, 32, 32)

        dec3 = self.upconv3(dec4) # (256, 64, 64)
        dec3 = torch.cat((dec3, enc3), dim=1) # (512, 64, 64)
        dec3 = self.decoder3(dec3) # (256, 64, 64)

        dec2 = self.upconv2(dec3) # (128, 128, 128)
        dec2 = torch.cat((dec2, enc2), dim=1) # (256, 128, 128)
        dec2 = self.decoder2(dec2) # (128, 128, 128)

        dec1 = self.upconv1(dec2) # (64, 256, 256)
        dec1 = torch.cat((dec1, enc1), dim=1) # (128, 256, 256)
        dec1 = self.decoder1(dec1) # (64, 256, 256)

        return self.final_conv(dec1) # (2, 256, 256)