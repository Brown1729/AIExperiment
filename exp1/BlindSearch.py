from numpy.matrixlib.defmatrix import matrix


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

    def graph_bfs(self, start_city: str, end_city: str) -> (list, list, int):
        """
        广度优先搜索（队列本身就是FIFO）
        要输出最终路径还得加一个并查集
        :param start_city: 起始城市名字
        :param end_city: 终止城市名字
        :return: 1：列表，代表路径，2：列表，搜索顺序, 3: 数字，代表路径长度
        """
        start = self.node_name.index(start_city)
        end = self.node_name.index(end_city)
        visited = [False] * self.n
        queue = []
        path = []
        search_path = []
        cost = 0
        uset = [None for i in range(self.n)]  # 并查集

        # 将起始点入队
        queue.append(start)
        visited[start] = True

        while queue:
            v = queue.pop(0)  # 取出队首元素
            search_path.append(v)  # 将其加入搜索顺序列表
            visited[v] = True  # 标记 v 已被访问
            if v == end:  # 如果到达终点
                break

            for i in range(self.n):
                if self.matrix[v][i] != self._MAX_COST_ and i != v and not visited[i]:  # 如果 v 到 i 有边且 i 未被访问过
                    queue.append(i)  # 将 i 入队
                    uset[i] = v  # 将 i 的父节点设为 v

        if uset[end] is None:
            return None, None, None

        # 从并查集反推
        path.append(end)
        while uset[path[-1]] is not None:
            cost += self.matrix[path[-1]][uset[path[-1]]]
            path.append(uset[path[-1]])

        return path[::-1], search_path, cost

    def graph_dfs(self, start_city: str, end_city: str) -> (list, list, int):
        """
        深度优先搜索（如果要LIFO的话，得用栈，不能递归，或者反向递归）
        要输出最终路径还得加一个并查集
        :param start_city: 起始城市名字
        :param end_city: 终止城市名字
        :return: 1：列表，代表路径，2：列表，搜索顺序, 3: 数字，代表路径长度
        """
        start = self.node_name.index(start_city)
        end = self.node_name.index(end_city)
        visited = [False] * self.n
        stack = []
        path = []
        search_path = []
        cost = 0
        uset = [None for i in range(self.n)]  # 并查集

        # 将起始点入栈
        stack.append(start)
        visited[start] = True

        while stack:
            v = stack.pop()  # 取出栈顶元素
            search_path.append(v)  # 将其加入搜索顺序列表
            visited[v] = True  # 标记 v 已被访问
            if v == end:  # 如果到达终点
                break

            for i in range(self.n):
                if self.matrix[v][i] != self._MAX_COST_ and i != v and not visited[i]:  # 如果 v 到 i 有边且 i 未被访问过
                    stack.append(i)  # 将 i 入栈
                    uset[i] = v  # 将 i 的父节点设为 v

        if uset[end] is None:
            return None, None, None

        # 从并查集反推
        path.append(end)
        while uset[path[-1]] is not None:
            cost += self.matrix[path[-1]][uset[path[-1]]]
            path.append(uset[path[-1]])

        return path[::-1], search_path, cost

    def graph_ucs(self, start_city: str, end_city: str) -> (list, list, int):
        """
        代价一致，如果代价相同就搜首个（类似dijkstra）
        还需要一个表存当前代价
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
        cost_list = [self._MAX_COST_ for i in range(self.n)]  # 代价表

        # 初始化
        cost_list[start] = 0

        while True:
            min_cost_vertex = cost_list.index(min(cost_list[i] for i in range(self.n) if not visited[i]))
            visited[min_cost_vertex] = True
            search_path.append(min_cost_vertex)
            if min_cost_vertex == end:
                break
            for v, c in enumerate(self.matrix[min_cost_vertex]):  # 找代价最小的
                if cost_list[v] > cost_list[min_cost_vertex] + c and c != self._MAX_COST_ and not visited[v]:
                    cost_list[v] = cost_list[min_cost_vertex] + c
                    uset[v] = min_cost_vertex

        if uset[end] is None:
            return None, None, None

        # 从并查集反推
        path.append(end)
        cost = cost_list[end]
        while uset[path[-1]] is not None:
            path.append(uset[path[-1]])

        return path[::-1], search_path, cost

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
        cost_list[start] = 0

        while True:
            min_cost_vertex = cost_list.index(min(cost_list[i] for i in range(self.n) if not visited[i]))
            visited[min_cost_vertex] = True
            search_path.append(min_cost_vertex)
            if min_cost_vertex == end:
                break
            for v, c in enumerate(self.matrix[min_cost_vertex]):  # 找代价最小的
                if cost_list[v] > heuristic_values[min_cost_vertex] and c != self._MAX_COST_ and not visited[v]:
                    cost_list[v] = heuristic_values[min_cost_vertex]
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
                if cost_list[v] > cost_list[min_cost_vertex] - heuristic_values[min_cost_vertex] + c + heuristic_values[v] and c != self._MAX_COST_ and not visited[v]:
                    cost_list[v] = cost_list[min_cost_vertex] - heuristic_values[min_cost_vertex] + c + heuristic_values[v]
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
    city_name_list = ["Arad", "Bucharest", "Craiova", "Dobreta", "Eforie", "Fagaras", "Giurgiu", "Hirsova", "Iasi",
                      "Lugoj", "Mehadia", "Neamt", "Oradea", "Pitesti", "Rimnicu Vilcea", "Sibiu", "Timisoara",
                      "Urziceni", "Vaslui", "Zerind"]
    graph = Graph(city_name_list)
    graph.add_edge_city("Arad", "Zerind", 75)
    graph.add_edge_city("Arad", "Sibiu", 140)
    graph.add_edge_city("Arad", "Timisoara", 118)
    graph.add_edge_city("Bucharest", "Urziceni", 85)
    graph.add_edge_city("Bucharest", "Giurgiu", 90)
    graph.add_edge_city("Bucharest", "Pitesti", 101)
    graph.add_edge_city("Bucharest", "Fagaras", 211)
    graph.add_edge_city("Craiova", "Dobreta", 120)
    graph.add_edge_city("Craiova", "Pitesti", 138)
    graph.add_edge_city("Craiova", "Rimnicu Vilcea", 146)
    graph.add_edge_city("Dobreta", "Mehadia", 75)
    graph.add_edge_city("Dobreta", "Craiova", 120)
    graph.add_edge_city("Eforie", "Hirsova", 86)
    graph.add_edge_city("Fagaras", "Sibiu", 99)
    graph.add_edge_city("Fagaras", "Bucharest", 211)
    graph.add_edge_city("Giurgiu", "Bucharest", 90)
    graph.add_edge_city("Hirsova", "Urziceni", 98)
    graph.add_edge_city("Hirsova", "Eforie", 86)
    graph.add_edge_city("Iasi", "Neamt", 87)
    graph.add_edge_city("Iasi", "Vaslui", 92)
    graph.add_edge_city("Lugoj", "Mehadia", 70)
    graph.add_edge_city("Lugoj", "Timisoara", 111)
    graph.add_edge_city("Mehadia", "Lugoj", 70)
    graph.add_edge_city("Mehadia", "Dobreta", 75)
    graph.add_edge_city("Neamt", "Iasi", 87)
    graph.add_edge_city("Oradea", "Zerind", 71)
    graph.add_edge_city("Oradea", "Sibiu", 151)
    graph.add_edge_city("Pitesti", "Rimnicu Vilcea", 97)
    graph.add_edge_city("Pitesti", "Craiova", 138)
    graph.add_edge_city("Pitesti", "Bucharest", 101)
    graph.add_edge_city("Rimnicu Vilcea", "Sibiu", 80)
    graph.add_edge_city("Rimnicu Vilcea", "Pitesti", 97)
    graph.add_edge_city("Rimnicu Vilcea", "Craiova", 146)
    graph.add_edge_city("Sibiu", "Fagaras", 99)
    graph.add_edge_city("Sibiu", "Oradea", 151)
    graph.add_edge_city("Sibiu", "Rimnicu Vilcea", 80)
    graph.add_edge_city("Sibiu", "Arad", 140)
    graph.add_edge_city("Timisoara", "Lugoj", 111)
    graph.add_edge_city("Timisoara", "Arad", 118)
    graph.add_edge_city("Urziceni", "Vaslui", 142)
    graph.add_edge_city("Urziceni", "Hirsova", 98)
    graph.add_edge_city("Urziceni", "Bucharest", 85)
    graph.add_edge_city("Vaslui", "Iasi", 92)
    graph.add_edge_city("Vaslui", "Urziceni", 142)
    graph.add_edge_city("Zerind", "Arad", 75)
    graph.add_edge_city("Zerind", "Oradea", 71)

    path, search_path, cost = graph.graph_bfs("Arad", "Bucharest")
    print("----------------------------------")
    print("BFS")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")

    path, search_path, cost = graph.graph_dfs("Arad", "Bucharest")
    print("----------------------------------")
    print("DFS")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")

    path, search_path, cost = graph.graph_ucs("Arad", "Bucharest")
    print("----------------------------------")
    print("UCS")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")

    dist_to_bucharest = [366, 0, 160, 242, 161, 178, 77, 151, 226, 244, 241, 234, 380, 98, 193, 253, 329, 80,
                         199, 374]
    path, search_path, cost = graph.graph_greedy("Arad", "Bucharest", dist_to_bucharest)
    print("----------------------------------")
    print("Greedy")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")

    path, search_path, cost = graph.graph_a_star("Arad", "Bucharest", dist_to_bucharest)
    print("----------------------------------")
    print("A*")
    print("路径：", graph.get_path_name(path))
    print("搜索顺序：", graph.get_path_name(search_path))
    print("路径长度：", cost)
    print("----------------------------------")
