import numpy as np
import matplotlib.pyplot as plt
import imageio


# 生成城市坐标
def generate_cities(n_cities, seed=0):
    np.random.seed(seed)
    cities = np.random.randint(0, 100, (n_cities, 2))
    cities = cities[np.argsort(cities[:, 0])]
    return cities


# 计算距离矩阵
def distance_matrix(cities):
    n_cities = cities.shape[0]
    dist_mat = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_mat[i, j] = np.linalg.norm(cities[i] - cities[j])
    return dist_mat


# 绘制结果
def plot_result(cities, best_path, model_name, names=True, s=200):
    plt.figure(dpi=150)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.scatter(cities[:, 0], cities[:, 1], label='cities', color='red', s=s)
    if names:
        for i, city in enumerate(cities):
            plt.annotate(str(i), (city[0], city[1]))

    for i in range(len(best_path)-1):
        city1 = cities[best_path[i]]
        city2 = cities[best_path[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]],
                 color='r', linewidth=1.5, linestyle='--')
    plt.plot([cities[best_path[-1]][0], cities[best_path[0]][0]], [cities[best_path[-1]][1],
             cities[best_path[0]][1]], color='r', linewidth=1.5, linestyle='--', label='best path')
    plt.title(f'TSP with {model_name}')
    plt.legend()
    plt.show()


# 生成最佳路线的变化gif
def generate_gif(cities, best_path_list, model_name, fps=10, names=True, s=200):
    images = []
    for i, best_path in enumerate(best_path_list):
        plt.figure(dpi=150)
        plt.grid(linestyle='-', linewidth=0.5)
        plt.scatter(cities[:, 0], cities[:, 1],
                    label='cities', color='red', s=s)
        if names:
            for j, city in enumerate(cities):
                plt.annotate(str(j), (city[0], city[1]))

        for j in range(len(best_path)-1):
            city1 = cities[best_path[j]]
            city2 = cities[best_path[j+1]]
            plt.plot([city1[0], city2[0]], [city1[1], city2[1]],
                     color='r', linewidth=1.5, linestyle='--')
        plt.plot([cities[best_path[-1]][0], cities[best_path[0]][0]], [cities[best_path[-1]][1],
                 cities[best_path[0]][1]], color='r', linewidth=1.5, linestyle='--', label='best path')
        plt.title(f'TSP with {model_name}, iteration {i}')
        plt.legend()
        plt.savefig(f'./images/1.png')
        plt.close()
        images.append(imageio.imread(f'./images/1.png'))
    imageio.mimsave(f'./images/{model_name}.gif', images, fps=fps)
