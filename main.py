from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy import NaN, ndarray
# Диапазон тензоров и тензоры по подобию имеющихся
# master_range = torch.range(start=1, step=3, end=333)

# like_range = torch.zeros_like(input=master_range)
# print(master_range, like_range)

# У тензора по дефолту разрядность 32 бита (float32)
# print(master_range.dtype)
# Можно приводить типы
# range_16 = master_range.type(torch.float16) # или torch.half
# При умножении тензоров разных разрядностей итог приводится к наибольшей
# range_multiple_32_16 = master_range*range_16
# print(range_16.dtype, range_multiple_32_16.dtype)
# Операции над тензорами: сложнеие вычитание умножение (почленное и матричное - краммеры гауссы, это все)ъ
# tensor = torch.rand(3, 3, 3)
# tensor2 = torch.rand(3, 3, 3)
# # Сравним по скорости алгоритмы матричного умножения
# start = time.perf_counter()
# value = 0
# for i in range(len(tensor)):
#     value += tensor[i] * tensor2[i]
# print(value)     
# end = time.perf_counter()
# print("Выполнение самописного умножения тензоров заняло {} c.".format(end-start))
# start1 = time.perf_counter()
# print(torch.matmul(tensor, tensor2))     
# end1 = time.perf_counter()
# print("Выполнение умножения тензоров пайторчем заняло {} c.".format(end1-start1))
# start2 = time.perf_counter()
# print(tensor @ tensor2)
# end2 = time.perf_counter()
# print("Выполнение умножения тензоров оператором @ заняло {} c.".format(end2-start2))
# 2 правила работы с матричным умножением:
# 1) У 2х тензоров должны совпадать внутренние измерения
# torch.rand(2,3) @ torch.rand(2,2) - так не будет работать
# torch.rand(2,3) @ torch.rand(3,10) - а так будет!
# 2) Результатом умножения будет тензор с внешними измерениями 2х исходных тензоров
# torch.rand(2,3) @ torch.rand(3,10) -> torch.Shape([2, 10])
# Чтобы подогнать тензоры по измерениям есть метод tensor.T (transpose), который делает из torch.rand(3,10) -> torch.rand(10,3)
# Щитаем среднее и медиану для выборки
# r = [-1,0,4,2,1,2]
# sum = sum(r)
# av = sum / len(r)
# def median(lst):
#     n = len(lst)
#     s = sorted(lst)
#     return (s[n//2-1]/2.0+s[n//2]/2.0, s[n//2])[n % 2] if n else None
# print(median(r))
# нормализация по макс и мин значениям признака
# a = np.array([1, 0, 5, 2, 2])
# b = np.vectorize(lambda x: (x - a.min())/(a.max() - a.min()))(a)
# print(b)

# # Создадим датафрейм с пропущенным значением признака Р для объекта А
# df = pd.DataFrame({'Молчание ягнят':[5,5,2,3], 
#                    'Титаник':[5,3,5,4], 
#                    'Матрица':[5,4,3,4], 
#                    'Гарри Поттер':[3,4,5,NaN]}, 
#                   index=['Вася', 'Петя', 'Маша', 'Саша'])
# print('df: \n', df)
# # Мин-макс нормализация
# #df = (df-df.min ())/(df.max ()-df.min ())
# #print('df_norm: \n', df)
# # Посчитаем метрики
# dict_metrics = {'Вася':[], 'Петя':[], 'Маша':[]}
# print('df.index: ', df.index[:-1])
# print('df.loc[Гарри Поттер]: \n ', df.loc[:]['Гарри Поттер'][-1])
# for i in df.index[:-1]:
#   dict_metrics[i].append(np.power((df.loc['Саша'][:-1]-df.loc[i][:-1]).pow(2).sum(), 0.5).round(2)) # считаем Евклидово расстояние
#   dict_metrics[i].append((df.loc['Саша'][:-1]-df.loc[i][:-1]).abs().sum()) # считаем Манхэттеновское расстояние
#   dict_metrics[i].append((df.loc['Саша'][:-1]-df.loc[i][:-1]).abs().max()) # считаем max-метрику
# print('dict_metrics: ',   dict_metrics) # {'A1': [1.41, 2.0, 1.0], 
#                                         #  'A2': [1.22, 2.0, 1.0], 
#                                         #  'A3': [0.87, 1.5, 0.5]}
# metrics = pd.DataFrame(dict_metrics, index=['Euclid', 'Manhatten', 'Max'])
# print('metrics: ', metrics)
# # Считаем варианты значений для каждой метрики
# dict_value = {'Euclid':[], 'Manhatten':[], 'Max':[]}
# for i in metrics.index:
#   norm_mul = (1/((1/metrics.loc[i]).sum())) # нормирующий множитель
#   similarity = ((df.loc[:]['Гарри Поттер'][:-1]/metrics.loc[i]).sum()) # значение признака * мера близости(=величина, обратно пропорциональная значению метрики)
#   value_P = (norm_mul*similarity).round(2)
#   dict_value[i].append(value_P)
#   print(f'значение признака Гарри Поттер для Саши по метрике {i}: {value_P}')
# Проверяем является ли значение выбросом методом квартилей
# Q25 =  2
# Q75 = 4
# BL = Q25 - 1.5 * (Q75 - Q25)
# BR = Q75 + 1.5 *(Q75 - Q25)
# print(np.arange(BL, BR))
# Проверяем является ли значение выбросом методом среднего значения, отклонения и медианы
# av = 10 # среднее значение
# S = 1.1 # отклонение
# m = 9 # медиана
# # так как медиана и среднее близки то выборка симметричная - коэффициент равен 3 (для нессиметричных было бы 5)
# BL =  av - 3 * S # левый край интервала
# BR = av + 3 * S # правый край интервала
# print(np.arange(BL, BR))
# Функция находит расстояния между объектами выборки методом Манхеттен 
# def manhet(list1, list2, list3, list4):
#     s = 0
#     for a, b, c, d in zip(list1, list2, list3, list4):
#         s += abs(a - b) + abs(a - c) + abs(a - d)
#     return s

# a = [1, 1, 0]
# b = [0, 2, -1]
# c = [2, 3, 1]
# d = [1, 0, 4]

# print('a:', manhet(a, b, c, d))
# print('b:', manhet(b, c, d, a))
# print('c:', manhet(c, b, a, d))
# print('d:', manhet(d, b, c, a))

# Реализация алгоритма к-средних (k-means) для кластеризации объектов по заранее известному количеству кластеров
# 1 Создадим пару массивов np.array с признаками
X = np.array([[4,2],[3,2],[1,-1],[-1,1],[0,4]])
# Создадим 2 центра масс для 2х кластеров
C = np.array([[2,3], [1,1]])
# Расстояние Манхэттена. Принимает 2 кортежа с признаками [Pа1, Pб1] объекта А
manhattan = lambda X, Y: np.abs(X - Y).sum(axis=1)
# Реализуем класс KMeans
class KMeans:
    # Инициализация.
    # n_clusters - желаемое число кластеров. 
    # centers - начальные значения центров кластеризации.
    # max_iter - максимальное число итераций алгоритма.
    # metric - метрика для расчета расстояний между центрами кластеризации и точек данных. (функция принимающая 2 числа и возвращающая число)
    def __init__(self, n_clusters, centers, max_iter, metric):
        self.n_clusters = n_clusters
        self.centers = centers
        self.max_iter = max_iter
        self.metric = metric
    @staticmethod
    def __euclid(X1, X2):
        # находим сумму квадратов расстояний между точками
        sum_sq = np.sum(np.square(X1 - X2))
        return sum_sq**0.5
    # Найти центр масс точек.
    @staticmethod
    def __mass_center(X: ndarray[int]):
        return X.mean(axis=0)
    # Кластеризуем
    def fit(self, X):
        distances = np.zeros((self.n_clusters, len(X))) # Расстояния от центров кластеризации до точек.
        self.n_iters = 0                                # Число итераций.
        for _ in range(self.max_iter):
            # Найти расстояния между центрами кластеризации и точками.
            for i in range(self.n_clusters):
                distances[i] = self.metric(self.centers[i], X)
            
            # Найти метки точек.
            self.labels = distances.argmin(axis=0)

            # Найти новые значения центров кластеризации.
            prev_centers = self.centers.copy()
            for i in np.unique(self.labels):
                self.centers[i] = self.__mass_center(X[self.labels == i])
            
            self.n_iters += 1 # Подсчитать число итераций алгоритма.
            # Если центры кластеризации практически не меняются, завершить алгоритм. 
            if self.__euclid(self.centers, prev_centers) < 10**-4:
                break
        return self

centers = np.random.randn(3, 2)
# Кластеризовать данные.
kmean = KMeans(n_clusters=2, centers=C, max_iter=100, metric=manhattan).fit(X)
print(f"Iterations: {kmean.n_iters}") # Вывести число итераций алгоритма.

# Отобразить кластеризованные данные и их центры масс. 
plt.scatter(X[:, 0], X[:, 1], c=kmean.labels)
plt.scatter(kmean.centers[:, 0], kmean.centers[:, 1], c='red')
plt.show()


