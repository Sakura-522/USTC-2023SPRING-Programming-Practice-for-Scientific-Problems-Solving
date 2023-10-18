import random
import numpy as np
import matplotlib.pyplot as plt


class GA:
    def __init__(self, cities, distance_matrix, pop_size, pc, pm, iter_max):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.iter_max = iter_max
        self.city_num = len(cities)
        self.pop = self.init_pop()
        self.best_path, self.best_distance = self.find_best()
        self.best_distance_list = [self.best_distance]
        self.best_path_list = [self.best_path]

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

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

    def init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            path = list(range(self.city_num))
            random.shuffle(path)
            pop.append(path)
        return pop

    def find_best(self):
        best_path, best_distance = None, np.inf
        for path in self.pop:
            path_distance = self.path_distance(path)
            if path_distance < best_distance:
                best_path, best_distance = path, path_distance
        return best_path, best_distance

    # 选择
    def selection(self, pop):
        group_num = 20  # 小组数
        group_size = len(pop) // group_num  # 每小组人数
        group_winner = self.pop_size // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        index = 0
        random.shuffle(pop)
        for _ in range(group_num):
            group = []
            for _ in range(group_size):
                # 随机组成小组
                # player = random.choice(population)
                player = pop[index]
                index += 1
                group.append(player)
                group.sort(key=lambda path: self.path_distance(path))
            # 取出获胜者
            winners += group[:group_winner]
        pop = winners
        return winners

    # 变异
    def mutation(self, genes):
        old_genes = genes.copy()
        index1 = random.randint(1, len(genes) - 3)
        index2 = random.randint(index1, len(genes) - 2)
        genes_mutate = old_genes[index1:index2]
        genes_mutate.reverse()
        genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        return genes

    # 交叉
    def crossover(self, pop):
        size = len(pop)
        random.shuffle(pop)
        for i in range(0, size-1, 2):
            genes1 = pop[i].copy()
            genes2 = pop[i+1].copy()
            index1 = random.randint(0, len(genes1)-3)
            index2 = random.randint(index1+1, len(genes1)-1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # 交叉
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2

            if random.random() < 0.5:
                pop.append(self.mutation(genes1))
                pop.append(self.mutation(genes2))
            else:
                pop.append(genes1)
                pop.append(genes2)
        return pop

    def genetic_algorithm(self):
        for _ in range(self.iter_max):
            # 交叉
            self.pop = self.crossover(self.pop)
            # 选择
            self.pop = self.selection(self.pop)
            # 计算最佳路径
            self.best_path, self.best_distance = self.find_best()
            self.best_distance_list.append(self.best_distance)
            self.best_path_list.append(self.best_path)

        return self.find_best()

    def plot_learning_curve(self):
        plt.figure(dpi=150)
        plt.plot(self.best_distance_list, color='red',
                 linewidth=1.5, linestyle='-.')
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.title("Searching Curve")
        plt.show()
