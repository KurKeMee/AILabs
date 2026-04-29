# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:36:26 2021

@author: AM4
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:13:16 2021

@author: AM4
"""

"""BERT Fine-Tuning Sentence Classification

Код базируется на https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP

# BERT Fine-Tuning Tutorial with PyTorch

By Chris McCormick and Nick Ryan

В коде используется библиотека [transformers](https://github.com/huggingface/transformers) 

Пред использованием необходимо установить пакеты:

conda install -c conda-forge transformers

"""

import torch
import numpy as np
import os
import pandas as pd


# Проверяем доступна ли GPU и задаем вычислительное устройство
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU')
    device = torch.device("cpu")


# В работе используется набор Russian Language Toxic Comments Dataset https://www.kaggle.com/blackmoon/russian-language-toxic-comments
# комментариев с сайтов Двач и Пикабу. 
# он опубликован  в 2019 году и содержит 14 412 комментариев
# 4 826 из них помечены как токсичные, а 9 586 — как нетоксичные

# Загрузка данных реализована на основе pandas dataframe
df = pd.read_csv("C:/Users/Victus/Desktop/Лабораторные/DeppNN/Лабораторные2026/7_BERT_classification/data/verb_communications.csv",
                 delimiter=';',
                 header=0,
                 names=['sentence', 'label'],
                 encoding='cp1251')

print('В наборе предложений: {:,}\n'.format(df.shape[0]))

# Пример
df.sample(10)

# Нас интересуют метки классов и сами предложения, на них мы будем обучать нашу сеть
sentences = df.sentence.values
labels = df.label.values


from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertForSequenceClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased',
    num_labels=3
)
model.to(device)


print(f'Размер словаря RuBERT: {tokenizer.vocab_size}')

max_length = 64
input_ids = np.zeros((len(sentences), max_length))

for s, i in zip(sentences, range(len(sentences))):
    enc_s = tokenizer.encode(
        str(s),
        add_special_tokens=True,
        padding='max_length',
        max_length=max_length,
        truncation=True
    )
    input_ids[i, :] = enc_s


# Создаем attention mask для виртуальных токенов
attention_masks = []

for s in input_ids:
    #   Если ID = 0, это виртуальный токен и маска для него 0.
    #   Если ID > 0, это реальный токен и маска для него 1.
    att_mask = [int(id_ > 0) for id_ in s]
    attention_masks.append(att_mask)


# Формируем тестовый и валидационный набор
from sklearn.model_selection import train_test_split

# разбиваем данные, метки классов
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.1)
# и маску
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1)

# все конвертируем в тензоры
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# теперь можно создавать Dataset и DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# размер батча придется уменьшить (если на GPU), т.к. за счет
# увеличившегося словаря выросла и модель 
batch_size = 4

# Пакуем в тренировочный предложения (ID), маску и метки классов
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# и в валидационный
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# теперь можно переходить к заданию модели

from transformers import BertForSequenceClassification, BertConfig
from torch.optim import AdamW


# Отправляем модель на GPU
if torch.cuda.is_available():
    model.cuda()

# Задаем оптимизатор
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # скорость обучения
                  eps = 1e-8,# специфический параметр, повышающий стабильность обучения
                )

from transformers import get_linear_schedule_with_warmup

# Количество эпох обучения
epochs = 6

# Шагов обучения = number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# scheduler - планировщик изменяющий скорость обучения
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.1 * total_steps), num_training_steps = total_steps)

# функция вычисления точности обучения
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime
import random

# Задаем seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Тут храним наши лоссы
loss_values = []
best_accuracy =0

# Цикл обучения будет состоять из обучения и валидации
for epoch_i in range(0, epochs):
    
    ################ Часть обучения #####################
    
    print("")
    print('Эпоха {:} из {:} '.format(epoch_i + 1, epochs))
    
    t0 = time.time()

    # потери за эпоху
    total_loss = 0
    
    # переключаем в режим обучения
    model.train()

    # пробегаем по батчам
    for step, batch in enumerate(train_dataloader):

       
        # Достаем из батча данные: предложения, маску и метки
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # обнуляем градиенты
        model.zero_grad()        

        # прямой проход
        outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
       
               # потери
        loss = outputs.loss
        total_loss += loss.item()

        # обратный проход
        loss.backward()

        # обрезаем градиенты до 1.0, чтобы предотвратить "взрыв градиентов"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # обновляем параметры модели
        optimizer.step()

        # изменяем скорость обучения
        scheduler.step()
        
         # диагностическую информацию выводим каждые 100 батчей
        if step % 10 == 0 and not step == 0:
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
            print(' Батч {:>4,} из {:>4,}. Затраченное время: {:}. Ошибка: {:}.'.format(step, len(train_dataloader), time_elapsed, loss))


    # средний loss 
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # сохраним для графика
    loss_values.append(avg_train_loss)

    print("")
    print(" Средний loss: {0:.2f}".format(avg_train_loss))
    print(" Обучение эпохи прошло за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))
        
    ################ Часть валидации #####################
    # Позволяет понять правильно ли мы учимся и учимся ли вообще

    print("\n Validation...")
    t0 = time.time()

    # модель в  evaluation режим
    model.eval()

    # диагностические переменные
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # пробегаем валидационный набор
    for batch in validation_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        
        # берем нужные данные из батча
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            # прямой проход
            outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
        
        # "logits" хранят вероятности классов похоже на softmax
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # считаем точность модели на валидации
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # суммарная точность
        eval_accuracy += tmp_eval_accuracy

        # сколько батчей прошло
        nb_eval_steps += 1

    val_accuracy = eval_accuracy / nb_eval_steps
    # результат валидации
    print("  Accuracy: {0:.2f}".format(val_accuracy))
    print("  Валидация прошла за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        model.save_pretrained('./my_rubert_verb_comm')
        tokenizer.save_pretrained('./my_rubert_verb_comm')



# Можно построить график обучения 
import matplotlib.pyplot as plt

plt.plot(loss_values, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# Загрузим предобученную модель
model.save_pretrained('./my_rubert_verb_comm')
tokenizer.save_pretrained('./my_rubert_verb_comm')


model.to(device)


new_test = [
    # Невербальная коммуникация (1)
    'Он театрально прикрыл рот ладонью, изображая ужас',
    'Девушка надула губы и отвернулась, показывая обиду',
    'Мужчина сложил руки на груди и откинулся в кресле',
    'Она взмахнула ресницами и томно улыбнулась кавалеру',
    'Ребёнок спрятал лицо в ладонях, играя в прятки',
    'Продавец указал взглядом на нужную полку',
    'Он галантно подал ей руку, помогая выйти из машины',
    'Студент поднял два пальца вверх, просясь выйти',
    'Официант кивнул, принимая заказ',
    'Учительница прижала палец к губам, успокаивая класс',
    'Спортсмен вскинул руки вверх, празднуя победу',
    'Полицейский вытянул руку, останавливая поток машин',
    'Девушка поправила шарф и улыбнулась своему отражению',
    'Мужчина нахмурился и покачал головой в ответ на предложение',
    'Она взяла его за руку и не отпускала всю прогулку',
    
    # Непроизвольные реакции (2)
    'У него потемнело в глазах от боли',
    'Она вздрогнула, когда прогремел гром',
    'Дыхание перехватило от ледяной воды',
    'Пальцы онемели на холоде без перчаток',
    'После удара в ухе зазвенел тонкий писк',
    'От страха его прошиб холодный пот',
    'Голос задрожал, когда он начал говорить',
    'У неё заслезились глаза от яркого солнца',
    'Во рту пересохло после долгой речи',
    'Колени задрожали сами собой от волнения',
    'Кровь прилила к лицу после неловкой шутки',
    'У него заложило уши в лифте',
    'От неожиданного окрика она выронила телефон',
    'Мурашки пробежали по телу от страшного рассказа',
    'На морозе щёки загорелись румянцем',
    
    # Вербальная коммуникация (0)
    'Мне завтра нужно сдать отчёт до обеда',
    'Ты не мог бы перезвонить мне через час',
    'Я люблю слушать шум дождя за окном',
    'По-моему, здесь очень красивое место',
    'Ты заметил, как изменился город за последний год',
    'Мне нужно серьёзно поговорить с тобой о нашем будущем',
    'Давай сегодня никуда не пойдём и останемся дома',
    'Как вы думаете, это правильное решение',
    'Она пообещала прийти ровно в восемь',
    'Я считаю его одним из лучших специалистов',
    'Вчера по телевизору показывали интересный фильм',
    'Не могли бы вы одолжить мне ручку на минуту',
    'Ты что, серьёзно собрался уволиться',
    'Мне нужно больше времени, чтобы всё взвесить',
    'Знаешь, я никогда не был в этом городе'
]

expected = [1]*15 + [2]*15 + [0]*15

classes = {
    0: 'Вербальная коммуникация',
    1: 'Невербальная коммуникация',
    2: 'Непроизвольная реакция'
}

correct = 0
results = []

for i, sentence in enumerate(new_test):
    enc_s = tokenizer.encode(
        sentence,
        add_special_tokens=True,
        padding='max_length',
        max_length=max_length,
        truncation=True
    )
    
    input_ids = torch.tensor([enc_s]).to(device)
    attention_mask = torch.tensor([[int(id_ > 0) for id_ in enc_s]]).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids.to(torch.long),
            token_type_ids=None,
            attention_mask=attention_mask.to(torch.long)
        )
    
    logits = outputs.logits.cpu().numpy()
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = np.argmax(logits, axis=1).flatten()[0]
    
    is_correct = pred == expected[i]
    if is_correct:
        correct += 1
    
    symbol = '+' if is_correct else '-'
    print(f'{sentence:<60} {classes[pred]:<30} {symbol}')
    results.append((sentence, pred, probs, expected[i]))

accuracy = correct / len(new_test) * 100
print(f'\nТочность на новых данных: {accuracy:.1f}% ({correct}/{len(new_test)})')
print(f'  Класс 0 (Вербальная коммуникация): {sum(1 for r in results if r[1] == r[3] and r[3] == 0)}/15')
print(f'  Класс 2 (Непроизвольные реакции): {sum(1 for r in results if r[1] == r[3] and r[3] == 2)}/15')
print(f'  Класс 1 (Невербальная коммуникация): {sum(1 for r in results if r[1] == r[3] and r[3] == 1)}/15')


    
    
while True:
    sentence = input("\nВведите предложение: ").strip()
    
    enc_s = tokenizer.encode(
         sentence,
         add_special_tokens=True,
         padding='max_length',
         max_length=max_length,
         truncation=True
    )
     
    input_ids = torch.tensor([enc_s]).to(device)
    attention_mask = torch.tensor([[int(id_ > 0) for id_ in enc_s]]).to(device)
     
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids.to(torch.long),
            token_type_ids=None,
            attention_mask=attention_mask.to(torch.long)
        )
     
    logits = outputs.logits.cpu().numpy()
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = np.argmax(logits, axis=1).flatten()[0]
    
    print(f'Класс: {pred} — {classes[pred]}')
    print(f'Вероятности: Неверб.={probs[1]:.2f} | Непроизв.={probs[2]:.2f} | Верб.={probs[0]:.2f}')
