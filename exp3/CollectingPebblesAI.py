"""
AI生成
"""
import math


class CollectingPebbles:
    def __init__(self, stones=10, max_take=3):
        """
        类的初始化函数
        :param stones: 石头个数
        :param max_take: 最大可选石头数
        """
        self.initial_stones = stones
        self.max_take = max_take
        self.current_stones = stones
        self.search_times_in_each_turn = 0
        self.game_tree = []  # 存储博弈树节点信息
        self.pruning_info = []  # 存储剪枝信息
        self.node_counter = 0  # 节点计数器

    def reset(self):
        """
        重置游戏
        :return:
        """
        self.current_stones = self.initial_stones
        self.game_tree = []
        self.pruning_info = []
        self.node_counter = 0

    def is_terminal(self, state):
        """
        判断当前状态是否为终端状态
        :param state:
        :return:
        """
        return state == 0

    def get_legal_moves(self, state):
        """
        获取当前状态下所有合法的移动
        :param state:
        :return:
        """
        return list(range(1, min(self.max_take, state) + 1))

    def make_move(self, state, move):
        """
        执行移动
        :param state:
        :param move:
        :return:
        """
        return state - move

    def evaluate(self, state, is_max):
        """
        评估函数：对于终端状态，赢返回1，输返回-1
        在MAX层，赢的话，结点赋值-1
        在MIN层，输的话，结点赋值1
        :param state:
        :param is_max:
        :return:
        """
        if state == 0:
            return -1 if is_max else 1
        return 0

    def min_max(self, state, depth, is_max, path="", parent_id=None):
        """
        MinMax算法实现
        :param state: 只在递归中进行计算，判断当前递归的支路是否需要停止
        :param depth: 递归深度
        :param is_max: 当前节点是最大还是最小，及MIN MAX区分
        :param path: 节点路径
        :param parent_id: 父节点ID
        :return: max_eval: 移动状态
                 best_move: 最佳移动
        """

        self.search_times_in_each_turn += 1
        node_id = self.node_counter
        self.node_counter += 1

        # 记录博弈树节点
        node_info = {
            'id': node_id,
            'parent_id': parent_id,
            'state': state,
            'depth': depth,
            'type': "MAX" if is_max else "MIN",
            'value': None,
            'move': None,
            'path': path,
            'pruned': False,
            'children': []
        }

        # 判断当前是否需要中止
        if self.is_terminal(state):
            eval_value = self.evaluate(state, is_max)
            node_info['value'] = eval_value
            self.game_tree.append(node_info)
            return eval_value, None

        if is_max:  # 如果是MAX层
            # 取下一层的最大值，作为本结点的值
            max_eval = -math.inf
            best_move = None  # 最佳移动步骤
            moves = self.get_legal_moves(state)

            for i, move in enumerate(moves):
                new_state = self.make_move(state, move)
                new_path = f"{path}/{i}" if path else str(i)
                eval, _ = self.min_max(new_state, depth + 1, False, new_path, node_id)
                if eval > max_eval:  # min结点选最小
                    max_eval = eval
                    best_move = move

            node_info['value'] = max_eval
            node_info['move'] = best_move
            self.game_tree.append(node_info)
            return max_eval, best_move
        else:  # 如果是MIN层
            # 取下一层最小值
            min_eval = math.inf
            best_move = None
            moves = self.get_legal_moves(state)

            for i, move in enumerate(moves):
                new_state = self.make_move(state, move)
                new_path = f"{path}/{i}" if path else str(i)
                eval, _ = self.min_max(new_state, depth + 1, True, new_path, node_id)
                if eval < min_eval:  # max结点选最大
                    min_eval = eval
                    best_move = move

            node_info['value'] = min_eval
            node_info['move'] = best_move
            self.game_tree.append(node_info)
            return min_eval, best_move

    def alpha_beta(self, state, depth, alpha, beta, is_max, path="", parent_id=None):
        """
        α-β剪枝算法实现
        :param state:
        :param depth:
        :param alpha:
        :param beta:
        :param is_max:
        :param path: 节点路径
        :param parent_id: 父节点ID
        :return:
        """

        self.search_times_in_each_turn += 1
        node_id = self.node_counter
        self.node_counter += 1

        # 记录博弈树节点
        node_info = {
            'id': node_id,
            'parent_id': parent_id,
            'state': state,
            'depth': depth,
            'type': "MAX" if is_max else "MIN",
            'value': None,
            'move': None,
            'path': path,
            'pruned': False,
            'alpha': alpha,
            'beta': beta,
            'children': []
        }

        # 判断当前状态是否中止
        if self.is_terminal(state):
            eval_value = self.evaluate(state, is_max)
            node_info['value'] = eval_value
            self.game_tree.append(node_info)
            return eval_value, None

        if is_max:
            max_eval = -math.inf
            best_move = None
            moves = self.get_legal_moves(state)

            for i, move in enumerate(moves):
                new_state = self.make_move(state, move)
                new_path = f"{path}/{i}" if path else str(i)

                # 检查是否会发生剪枝
                if beta <= alpha:
                    # 创建被剪枝的节点
                    pruned_node_id = self.node_counter
                    self.node_counter += 1
                    pruned_node_info = {
                        'id': pruned_node_id,
                        'parent_id': node_id,
                        'state': new_state,
                        'depth': depth + 1,
                        'type': "MIN",
                        'value': "×",
                        'move': move,
                        'path': new_path,
                        'pruned': True,
                        'alpha': alpha,
                        'beta': beta,
                        'children': []
                    }
                    self.game_tree.append(pruned_node_info)

                    pruning_msg = f"β剪枝发生在深度{depth}, 节点{new_path}, alpha={alpha}, beta={beta}"
                    self.pruning_info.append(pruning_msg)
                    continue  # 跳过这个分支

                eval, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, False, new_path, node_id)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)  # alpha 保存下界，MAX 的最佳值

            node_info['value'] = max_eval
            node_info['move'] = best_move
            node_info['alpha'] = alpha
            node_info['beta'] = beta
            self.game_tree.append(node_info)
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            moves = self.get_legal_moves(state)

            for i, move in enumerate(moves):
                new_state = self.make_move(state, move)
                new_path = f"{path}/{i}" if path else str(i)

                # 检查是否会发生剪枝
                if beta <= alpha:
                    # 创建被剪枝的节点
                    pruned_node_id = self.node_counter
                    self.node_counter += 1
                    pruned_node_info = {
                        'id': pruned_node_id,
                        'parent_id': node_id,
                        'state': new_state,
                        'depth': depth + 1,
                        'type': "MAX",
                        'value': "×",
                        'move': move,
                        'path': new_path,
                        'pruned': True,
                        'alpha': alpha,
                        'beta': beta,
                        'children': []
                    }
                    self.game_tree.append(pruned_node_info)

                    pruning_msg = f"α剪枝发生在深度{depth}, 节点{new_path}, alpha={alpha}, beta={beta}"
                    self.pruning_info.append(pruning_msg)
                    continue  # 跳过这个分支

                eval, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, True, new_path, node_id)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)  # beta 保存上界，MIN 的最佳值

            node_info['value'] = min_eval
            node_info['move'] = best_move
            node_info['alpha'] = alpha
            node_info['beta'] = beta
            self.game_tree.append(node_info)
            return min_eval, best_move

    def build_tree_structure(self):
        """构建树形结构"""
        # 首先按ID排序
        nodes = sorted(self.game_tree, key=lambda x: x['id'])

        # 构建父子关系
        node_dict = {node['id']: node for node in nodes}
        for node in nodes:
            if node['parent_id'] is not None and node['parent_id'] in node_dict:
                parent = node_dict[node['parent_id']]
                parent['children'].append(node)

        # 找到根节点
        root_nodes = [node for node in nodes if node['parent_id'] is None]
        return root_nodes[0] if root_nodes else None

    def print_tree_node(self, node, prefix="", is_last=True, algorithm='minmax'):
        """递归打印树节点"""
        # 构建连接符
        connector = "└── " if is_last else "├── "

        # 构建节点显示信息
        if algorithm == 'minmax':
            node_info = f"{node['type']}(s={node['state']}, v={node['value']})"
        else:  # alphabeta
            if node['pruned']:
                node_info = f"{node['type']}(s={node['state']}, v=×) [PRUNED]"
            else:
                node_info = f"{node['type']}(s={node['state']}, v={node['value']}, α={node.get('alpha', 'N/A')}, β={node.get('beta', 'N/A')})"

        print(prefix + connector + node_info)

        # 更新前缀
        new_prefix = prefix + ("    " if is_last else "│   ")

        # 递归打印子节点
        for i, child in enumerate(node['children']):
            is_last_child = (i == len(node['children']) - 1)
            self.print_tree_node(child, new_prefix, is_last_child, algorithm)

    def print_game_tree(self, algorithm):
        """打印博弈树"""
        print(f"\n{'='*60}")
        print(f"{algorithm.upper()}算法博弈树")
        print(f"{'='*60}")

        root = self.build_tree_structure()
        if root:
            self.print_tree_node(root, "", True, algorithm)
        else:
            print("无法构建树结构")

        print(f"\n总搜索节点数: {len(self.game_tree)}")
        print(f"搜索次数: {self.search_times_in_each_turn}")

    def print_pruning_info(self):
        """打印剪枝信息"""
        if self.pruning_info:
            print(f"\n{'='*40}")
            print("α-β剪枝信息")
            print(f"{'='*40}")
            for i, info in enumerate(self.pruning_info, 1):
                print(f"{i}. {info}")
            print(f"总共发生了 {len(self.pruning_info)} 次剪枝")
        else:
            print("\n本次搜索中没有发生剪枝")

    def get_ai_move(self, algorithm='alphabeta', verbose=False):
        """
        获取AI的移动
        :param algorithm: 算法类型
        :param verbose: 是否显示详细信息
        :return:
        """

        self.search_times_in_each_turn = 0
        self.game_tree = []
        self.pruning_info = []
        self.node_counter = 0

        if algorithm == 'minmax':
            _, move = self.min_max(self.current_stones, 0, True)  # AI 思考是以自己为 MAX 的，即使从全局来看属于 MIN
        else:  # alphabeta
            _, move = self.alpha_beta(self.current_stones, 0, -math.inf, math.inf, True)

        print(f"AI 搜索了 {self.search_times_in_each_turn} 次")

        if verbose:
            self.print_game_tree(algorithm)
            if algorithm == 'alphabeta':
                self.print_pruning_info()

        return move

    def player_move(self, move):
        """
        玩家移动
        :param move:
        :return:
        """
        # 非法移动
        if move < 1 or move > self.max_take or move > self.current_stones:
            return False
        self.current_stones -= move
        return True

    def display_state(self):
        """显示当前游戏状态"""
        print(f"\n当前石子数: {self.current_stones}")
        print("石子: " + "● " * self.current_stones)


if "__main__" == __name__:
    STONE_NUM = 4
    game = CollectingPebbles(STONE_NUM, 3)

    print("=" * 50)
    print("           石子游戏")
    print("=" * 50)
    print("规则:")
    print(f"- 初始有{STONE_NUM}颗石子")
    print("- 每次可以取1-3颗石子")
    print("- 取走最后一颗石子的玩家输")
    print("=" * 50)

    while True:
        print("\n选择游戏模式:")
        print("1. 玩家先手")
        print("2. AI先手")
        print("3. 退出游戏")

        choice = input("请选择(1-3): ").strip()

        if choice == '3':
            print("游戏结束！")
            break
        elif choice in ['1', '2']:
            algorithm_choice = input("选择AI算法 (1: MinMax, 2: AlphaBeta): ").strip()
            algorithm = 'minmax' if algorithm_choice == '1' else 'alphabeta'

            verbose_choice = input("是否显示算法详细信息? (y/n): ").strip().lower()
            verbose = verbose_choice == 'y'

            game.reset()
            player_turn = (choice == '1')  # True表示玩家先手

            print(f"\n游戏开始! {'玩家' if player_turn else 'AI'}先手")
            print(f"使用算法: {algorithm}")

            while game.current_stones > 0:
                game.display_state()

                if player_turn:
                    # 玩家回合
                    while True:
                        try:
                            move = int(input(f"\n请取石子(1-{min(3, game.current_stones)}): "))
                            if game.player_move(move):
                                break
                            else:
                                print("无效的移动! 请重新输入。")
                        except ValueError:
                            print("请输入数字!")

                    if game.current_stones == 0:
                        print("\n你取走了最后一颗石子，你赢了！")
                        break
                else:
                    # AI回合
                    print("\nAI思考中...")
                    move = game.get_ai_move(algorithm, verbose)
                    game.current_stones -= move
                    print(f"AI取走了 {move} 颗石子")

                    if game.current_stones == 0:
                        print("\nAI取走了最后一颗石子，AI赢了！")
                        break

                player_turn = not player_turn  # 切换回合

            # 询问是否再来一局
            play_again = input("\n是否再来一局? (y/n): ").strip().lower()
            if play_again != 'y':
                print("游戏结束！")
                break