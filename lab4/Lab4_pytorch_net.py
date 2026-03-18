import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn 
import pandas as pd

###############################################################################
###############################################################################
# Теперь посмотрим что изменится в структуре нейронной сети, если нам нужно решить задачу регрессии
# Задача регрессии заключается в предсказании значений одной переменной
# по значениям другой (других).  
# От задачи классификации отличается тем, что выходные значения нейронной сети не 
# ограничиваются значениями меток классов (0 или 1), а могут лежать в любом 
# диапазоне чисел.
# Примерами такой задачи можгут быть предсказание цен на жилье, курсов валют или акций,
# количества выпадающих осадков или потребления электроэнергии.

# Рассмотрим задачу предсказания прочности бетона (измеряется в мегапаскалях)
df = pd.read_csv('C:\\Users\\Victus\\Desktop\\Лабораторные\\ИИ\\lab4\\dataset_simple.csv')


X = torch.Tensor(df.iloc[:, 0:2].values) # выделяем признаки (независимые переменные)
y = torch.Tensor(df.iloc[:, 2].values)  #  предсказываемая переменная, ее берем из последнего столбца


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, [1]].values, marker='o')

# Чтобы выходные значения сети лежали в произвольном диапазоне,
# выходной нейрон не должен иметь функции активации или 
# фуннкция активации должна иметь область значений от -бесконечность до +бесконечность

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size) # просто сумматор
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = 2 # количество признаков задачи 
hiddenSizes = 3   #  число нейронов скрытого слоя 
outputSize = 1 # число нейронов выходного слоя

net = NNet_regression(inputSize,hiddenSizes,outputSize)

# В задачах регрессии чаще используется способ вычисления ошибки как разница квадратов
# как усредненная разница квадратов правильного и предсказанного значений (MSE)
# или усредненный модуль разницы значений (MAE)
lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred.squeeze(), y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    

# Load data
df = pd.read_csv('C:\\Users\\Victus\\Desktop\\Лабораторные\\ИИ\\lab4\\dataset_simple.csv')

X = torch.Tensor(df.iloc[:, 0:2].values) # input features (96x1 tensor)
y = torch.Tensor(df.iloc[:, 2].values)  # target values

# Plot original data
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Не купит')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Купит')
plt.title('Original Data')
plt.xlabel('Feature')
plt.ylabel('Target')

class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        return self.layers(X)

# Correct network parameters
inputSize = 2    # Only one input feature
hiddenSizes = 3   # Hidden layer neurons
outputSize = 1    # One output value

net = NNet_regression(inputSize, hiddenSizes, outputSize)
lossFn = nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Training loop
epochs = 100
for i in range(epochs):
    pred = net(X)           # Forward pass
    loss = lossFn(pred, y)  # Calculate loss
    
    optimizer.zero_grad()   # Zero gradients
    loss.backward()         # Backpropagation
    optimizer.step()        # Update weights
    
    if i % 10 == 0:
        print(f'Epoch {i+1}, Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    pred = net(X)
    print('\nSample predictions:')
    print(pred[:10])
    
    err = torch.mean(abs(y - pred))
    print(f'\nMAE: {err.item():.4f} MPa')
































































































# Пасхалка, кто найдет и сможет объяснить, тому +
# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])

# y = df.iloc[:, -1].values

# w = np.linalg.inv(X.T @ X) @ X.T @ y

# predicted = X @ w

# print(predicted)


























