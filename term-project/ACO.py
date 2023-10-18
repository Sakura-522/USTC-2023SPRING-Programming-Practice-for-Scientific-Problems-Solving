import random
import numpy as np
import matplotlib.pyplot as plt
import copy


class Ant:
    def __init__(self, ID, city_num, distance_matrix, pheromone_matrix, alpha, beta):
        self.ID = ID                             # ID
        self.city_num = city_num                 # 城市数量
        self.distance_matrix = distance_matrix     # 距离矩阵
        self.pheromone_matrix = pheromone_matrix   # 信息素矩阵
        self.alpha = alpha                       # 信息素重要程度
        self.beta = beta                         # 启发式因子重要程度
        self.__set_ant()                         # 初始化出生点

    def __set_ant(self):
        self.path = []               # 当前蚂蚁的路径
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for _ in range(self.city_num)]  # 探索城市的状态

        city_index = random.randint(0, self.city_num-1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for _ in range(self.city_num)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(self.city_num):
            if self.open_table_city[i]:
                # 计算概率：与信息素浓度成正比，与距离成反比
                select_citys_prob[i] = pow(self.pheromone_matrix[self.current_city][i], self.alpha) * pow(
                    (1.0/self.distance_matrix[self.current_city][i]), self.beta)
                total_prob += select_citys_prob[i]

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, self.city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += self.distance_matrix[start][end]

        # 回路
        end = self.path[0]
        temp_distance += self.distance_matrix[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__set_ant()

        # 搜索路径，遍历完所有城市为止
        while self.move_count < self.city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()


class ACO(object):
    def __init__(self, city_num, distance, ant_num=50, alpha=1.0, beta=2.0, rho=0.5, Q=1, iter_max=200):

        self.city_num = city_num
        self.ant_num = ant_num
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.iter_max = iter_max

        # 获得距离矩阵
        self.distance_matrix = distance

        self.best_distance_list = []
        self.best_path_list = []

    def set_seed(self, seed):
        random.seed(seed)

    # 初始化
    def new(self):

        # 初始城市之间的信息素
        self.pheromone_matrix = np.ones((self.city_num, self.city_num))
        self.ants = [Ant(ID, self.city_num, self.distance_matrix, self.pheromone_matrix,
                         self.alpha, self.beta) for ID in range(self.ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1, self.city_num, self.distance_matrix,
                            self.pheromone_matrix, self.alpha, self.beta)  # 初始最优解
        self.best_ant.total_distance = 1 << 31               # 初始最大距离
        self.iter = 1                                        # 初始化迭代次数

    # 更新信息素
    def __update_pheromone_gragh(self):
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = np.zeros((self.city_num, self.city_num))
        for ant in self.ants:
            for i in range(1, self.city_num):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += self.Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone_matrix[i][j] = self.pheromone_matrix[i][j] * \
                    self.rho + temp_pheromone[i][j]

    # 开始搜索

    def search_path(self):
        for ant in self.ants:
            # 搜索一条路径
            ant.search_path()
            # 与当前最优蚂蚁比较
            if ant.total_distance < self.best_ant.total_distance:
                # 更新最优解
                self.best_ant = copy.deepcopy(ant)
        # 更新信息素
        self.__update_pheromone_gragh()
        # 保存最优解
        self.best_distance_list.append(self.best_ant.total_distance)
        self.best_path_list.append(self.best_ant.path)
        self.iter += 1

    def ant_colony_algorithm(self):
        # 初始化
        self.new()
        # 迭代搜索
        while self.iter < self.iter_max:
            self.search_path()

        # 返回最优解
        return self.best_ant.path, self.best_ant.total_distance

    def plot_learning_curve(self):
        plt.figure(dpi=150)
        # plt.style.use('seaborn')
        plt.plot(self.best_distance_list, color='red', linewidth=1.5, linestyle='-.')
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.title("Searching Curve")
        plt.show()
