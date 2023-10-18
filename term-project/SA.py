import random
import numpy as np
import matplotlib.pyplot as plt


class SA:
    def __init__(self, cities, distance_matrix, initial_temperature, cooling_rate, num_iterations):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.best_path = None
        self.best_distance = None
        self.best_distance_list = []
        self.best_path_list = []

    # 固定随机数种子，以便复现结果
    def set_seed(self, seed):
        random.seed(seed)

    # 计算当前路径的总距离
    def path_distance(self, path):
        path_distance = 0.0
        for i in range(1, len(self.cities)):
            start, end = path[i], path[i-1]
            path_distance += self.distance_matrix[start][end]

        # 回路
        end = path[0]
        path_distance += self.distance_matrix[start][end]
        return path_distance

    # 模拟退火算法
    def simulated_annealing(self):
        num_cities = len(self.cities)
        # 随机初始化路径
        current_path = random.sample(range(num_cities), num_cities)
        best_path = current_path.copy()
        best_distance = self.path_distance(best_path)

        temperature = self.initial_temperature

        for _ in range(self.num_iterations):
            # 随机选择两个城市进行交换
            i, j = random.sample(range(num_cities), 2)
            new_path = current_path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]

            # 计算新路径的距离
            new_distance = self.path_distance(new_path)

            # 接受新路径的概率
            acceptance_prob = np.exp(
                (best_distance - new_distance) / temperature)

            # 根据概率决定是否接受新路径
            if acceptance_prob > random.random():
                current_path = new_path.copy()

            # 更新最佳路径
            if new_distance < best_distance:
                best_path = new_path.copy()
                best_distance = new_distance
                
            self.best_distance_list.append(best_distance)
            self.best_path_list.append(best_path)

            # 降低温度
            temperature *= self.cooling_rate

        self.best_path = best_path
        self.best_distance = best_distance
        return best_path, best_distance

    def plot_learning_curve(self):
        plt.figure(dpi=150)
        # plt.style.use('seaborn')
        plt.plot(self.best_distance_list, color='red',
                 linewidth=1.5, linestyle='-.')
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.title("Searching Curve")
        plt.show()
