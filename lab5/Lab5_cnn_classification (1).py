# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:05:20 2021

@author: AM4
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Так как сеть, которую мы планируем взять за базу натренирована на изображениях 
# определенного размера, то наши изображения необходимо к ним преобразовать
data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])


# Пересоздадим датасеты с учетом новых размеров и нормировки яркости
train_dataset = torchvision.datasets.ImageFolder(root='C:\\Users\\Victus\\Desktop\\Лабораторные\\ИИ\\lab5\\train',
                                                 transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='C:\\Users\\Victus\\Desktop\\Лабораторные\\ИИ\\lab5\\test',
                                             transform=data_transforms)

class_names = train_dataset.classes

test_dataset.classes
batch_size = 15

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=2)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=2) 


# В качестве донора возьмем преобученную на ImageNet наборе сеть AlexNet
# Список доступных предобученных сетей можно посмотреть тут https://pytorch.org/vision/main/models.html
net = torchvision.models.alexnet(pretrained=True)

# можно посмотреть структуру этой сети
print(net)

# Так как веса feature_extractor уже обучены, нам нужно их "заморозить", чтобы 
# быстрее научился наш классификатор
#  для этого отключаем у всех слоев (включая слои feature_extractor-а) градиенты
for param in net.parameters():
    param.requires_grad = False


# Выходной слой AlexNet содержит 1000 нейронов (по количеству классов в ImageNet).
# Нам нужно его заменить на слой, содержащий только 2 класса.

num_classes = 3

new_classifier = net.classifier[:-1] # берем все слой классификатора кроме последнего
new_classifier.add_module('fc',nn.Linear(4096,num_classes))# добавляем последним слой с двумя нейронами на выходе
net.classifier = new_classifier # меняем классификатор сети

net = net.to(device)

# проверим эффективность новой сети
correct_predictions = 0
num_test_samples = len(test_dataset)


#до этого момента копируем и вставляем, потом остальную часть

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# явно требуется обучение


# Перейдем к обучению.
# Зададим количество эпох обучения, функционал потерь и оптимизатор.
num_epochs = 2
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
import time
t = time.time()
save_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        save_loss.append(loss.item())
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

# Посмотрим как уменьшался loss в процессе обучения
plt.figure()
plt.plot(save_loss)

# Еще раз посчитаем точность нашей модели
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')
# уже лучше


# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)
