import math

class Graph:
    _MAX_COST_ = 2 ** 31 - 1

    def __init__(self, node_name):
        """
        邻接矩阵
        """
        self.node_name = node_name
        self.n = len(self.node_name)

        self.matrix = [[self._MAX_COST_ for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            self.matrix[i][i] = 0  # 自己到自己置零

    def add_edge(self, u, v, cost) -> None:
        """
        添加 u 到 v 的边
        :param u: 起始节点
        :param v: 目标节点
        :param cost: 边的权重或成本
        :return: 无返回值
        """
        self.matrix[u][v] = cost  # 在邻接矩阵中将 u 到 v 的边设置为给定的成本值

    def add_edge_city(self, city1, city2, cost):
        self.add_edge(self.node_name.index(city1), self.node_name.index(city2), cost)

    def get_path_name(self, path):
        return [self.node_name[i] for i in path]

    def graph_greedy(self, start_city: str, end_city: str, heuristic_values: list) -> (list, list, int):
        """
        贪婪搜索，
        :param start_city: 起始城市名字
        :param end_city: 终止城市名字
        :return: 1：列表，代表路径，2：列表，搜索顺序, 3: 数字，代表路径长度
        """
        start = self.node_name.index(start_city)
        end = self.node_name.index(end_city)
        visited = [False] * self.n
        path = []
        search_path = []
        cost = 0
        uset = [None for i in range(self.n)]  # 并查集
        cost_list = [self._MAX_COST_ for i in range(self.n)]  # 启发信息表

        # 初始化
        cost_list[start] = heuristic_values[start]

        while True:
            min_cost_vertex = cost_list.index(min(cost_list[i] for i in range(self.n) if not visited[i]))
            visited[min_cost_vertex] = True
            search_path.append(min_cost_vertex)
            if min_cost_vertex == end:
                break
            for v, c in enumerate(self.matrix[min_cost_vertex]):  # 找代价最小的
                if cost_list[v] > heuristic_values[min_cost_vertex] and c != self._MAX_COST_ and not visited[v]:
                    cost_list[v] = heuristic_values[v]
                    uset[v] = min_cost_vertex

        if uset[end] is None:
            return None, None, None

        # 从并查集反推
        path.append(end)
        while uset[path[-1]] is not None:
            cost += self.matrix[path[-1]][uset[path[-1]]]
            path.append(uset[path[-1]])

        return path[::-1], search_path, cost

    def graph_a_star(self, start_city: str, end_city: str, heuristic_values: list) -> (list, list, int):
        """
        A*搜索，
        :param start_city: 起始城市名字
        :param end_city: 终止城市名字
        :return: 1：列表，代表路径，2：列表，搜索顺序, 3: 数字，代表路径长度
        """
        start = self.node_name.index(start_city)
        end = self.node_name.index(end_city)
        visited = [False] * self.n
        path = []
        search_path = []
        cost = 0
        uset = [None for i in range(self.n)]  # 并查集
        cost_list = [self._MAX_COST_ for i in range(self.n)]  # 启发信息表

        # 初始化
        cost_list[start] = 0 + heuristic_values[start]

        while True:
            min_cost_vertex = cost_list.index(min(cost_list[i] for i in range(self.n) if not visited[i]))
            visited[min_cost_vertex] = True
            search_path.append(min_cost_vertex)
            if min_cost_vertex == end:
                break
            for v, c in enumerate(self.matrix[min_cost_vertex]):  # 更新相邻节点代价
                # 这里代价评估需要算清除
                if cost_list[v] > cost_list[min_cost_vertex] - heuristic_values[min_cost_vertex] + c + heuristic_values[
                    v] and c != self._MAX_COST_ and not visited[v]:
                    cost_list[v] = cost_list[min_cost_vertex] - heuristic_values[min_cost_vertex] + c + \
                                   heuristic_values[v]
                    uset[v] = min_cost_vertex

        if uset[end] is None:
            return None, None, None

        # 从并查集反推
        path.append(end)
        cost = cost_list[end] - heuristic_values[end]
        while uset[path[-1]] is not None:
            path.append(uset[path[-1]])

        return path[::-1], search_path, cost


if __name__ == '__main__':
    city_name_list = ['Beijing', 'Hefei', 'Jinan', 'Nanjing', 'Shanghai', 'Shijiazhuang', 'Tianjin', "Wuhan"]
    print(city_name_list)
    graph = Graph(city_name_list)

    graph.add_edge_city("Beijing", "Shijiazhuang", 290)
    graph.add_edge_city("Beijing", "Tianjin", 120)
    graph.add_edge_city("Hefei", "Jinan", 580)
    graph.add_edge_city("Hefei", "Nanjing", 160)
    graph.add_edge_city("Hefei", "Wuhan", 290)
    graph.add_edge_city("Jinan", "Hefei", 580)
    graph.add_edge_city("Jinan", "Shijiazhuang", 330)
    graph.add_edge_city("Jinan", "Tianjin", 350)
    graph.add_edge_city("Nanjing", "Hefei", 160)
    graph.add_edge_city("Nanjing", "Shanghai", 300)
    graph.add_edge_city("Shanghai", "Nanjing", 300)
    graph.add_edge_city("Shijiazhuang", "Beijing", 290)
    graph.add_edge_city("Shijiazhuang", "Jinan", 330)
    graph.add_edge_city("Shijiazhuang", "Wuhan", 350)
    graph.add_edge_city("Tianjin", "Beijing", 120)
    graph.add_edge_city("Tianjin", "Jinan", 350)
    graph.add_edge_city("Wuhan", "Hefei", 290)
    graph.add_edge_city("Wuhan", "Shijiazhuang", 350)

    start_city = "Shijiazhuang"
    end_city = "Shanghai"
    city_position = {"Beijing": (116.4, 39.9), "Hefei": (117.2, 31.8), "Jinan": (117.0, 36.6), "Nanjing": (118.7, 32.0),
                     "Shanghai": (121.4, 31.2), "Shijiazhuang": (114.5, 38.0), "Tianjin": (117.2, 39.1),
                     "Wuhan": (114.3, 30.5)}
    dist_to_end_city = [math.hypot(city_position[end_city][0] - i[0], city_position[end_city][1] - i[1]) for i in
                        city_position.values()]

    print(dist_to_end_city)
    path, search_path, cost = graph.graph_greedy("Shijiazhuang", "Shanghai", dist_to_end_city)
    print("----------------------------------")
    print("Greedy")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")

    path, search_path, cost = graph.graph_a_star("Shijiazhuang", "Shanghai", dist_to_end_city)
    print("----------------------------------")
    print("A*")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")
