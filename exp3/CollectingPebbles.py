"""
盘子中有10颗石子，两个玩家轮流从中取出。至少取1颗，至多取3颗。不能继续操作的玩家输
MinMax算法
α-β 剪枝算法
和你自己玩（命令行交互界面）
"""
"""
一、组织方式
使用递归函数，参数为当前盘面状态，返回值为当前盘面状态下的最优得分
二、几个问题
1.如何组织数据结构？
2.如何判断当前盘面状态？
3.如何看算法是否真的运行成功？是否需要用控制台专门输出？
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
        self.prune_times_in_each_turn = 0

    def get_current_stones(self) -> int:
        """
        获取当前石头数
        :return:
        """
        return self.current_stones

    def reset(self):
        """
        重置游戏
        :return:
        """
        self.current_stones = self.initial_stones

    def is_terminal(self, state):
        """
        判断当前状态是否为终止状态
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

    def min_max(self, state, depth, is_max):
        """
        MinMax算法实现
        :param state: 只在递归中进行计算，判断当前递归的支路是否需要停止
        :param depth: 递归深度
        :param is_max: 当前节点是最大还是最小，及MIN MAX区分
        :return: max_eval: 移动状态
                 best_move: 最佳移动
        """

        self.search_times_in_each_turn += 1

        # 判断当前是否需要中止
        if self.is_terminal(state):
            return self.evaluate(state, is_max), None

        if is_max:  # 如果是MAX层
            # 取下一层的最大值，作为本结点的值
            max_eval = -math.inf
            best_move = None  # 最佳移动步骤
            for move in self.get_legal_moves(state):
                new_state = self.make_move(state, move)
                eval, _ = self.min_max(new_state, depth + 1, False)  # 往下递归，深度加一
                if eval > max_eval:  # min结点选最小
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:  # 如果是MIN层
            # 取下一层最小值
            min_eval = math.inf
            best_move = None
            for move in self.get_legal_moves(state):
                new_state = self.make_move(state, move)
                eval, _ = self.min_max(new_state, depth + 1, True)  # 往下递归，深度加一
                if eval < min_eval:  # max结点选最大
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def alpha_beta(self, state, depth, alpha, beta, is_max):
        """
        α-β剪枝算法实现
        :param state:
        :param depth:
        :param alpha:
        :param beta:
        :param is_max:
        :return:
        """

        self.search_times_in_each_turn += 1

        # 判断当前状态是否中止
        if self.is_terminal(state):
            return self.evaluate(state, is_max), None

        if is_max:
            max_eval = -math.inf
            best_move = None
            # 取下一层最大值，如果满足剪枝条件，则执行剪枝
            for move in self.get_legal_moves(state):
                new_state = self.make_move(state, move)
                eval, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)  # alpha 保存下界，MAX 的最佳值
                if beta <= alpha:  # 满足剪枝条件 beta 小于 alpha
                    # print(f"剪枝发生在{depth}, 此时alpha={alpha}，beta={beta}，是否为MAX；{is_max}")
                    self.prune_times_in_each_turn += 1
                    break  # β剪枝
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            # 取下一层最小值，同上
            for move in self.get_legal_moves(state):
                new_state = self.make_move(state, move)
                eval, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)  # beta 保存上界，MIN 的最佳值
                if beta <= alpha:  # 满足剪枝条件 beta 小于 alpha
                    # print(f"剪枝发生在{depth}, 此时alpha={alpha}，beta={beta}，是否为MAX；{is_max}")
                    self.prune_times_in_each_turn += 1
                    break  # α剪枝
            return min_eval, best_move

    def get_ai_move(self, algorithm='alphabeta'):
        """
        获取AI的移动
        :param algorithm:
        :return:
        """

        self.search_times_in_each_turn = 0
        self.prune_times_in_each_turn = 0

        if algorithm == 'minmax':
            _, move = self.min_max(self.current_stones, 0, True)  # AI 思考是以自己为 MAX 的，即使从全局来看属于 MIN
        else:  # alphabeta
            _, move = self.alpha_beta(self.current_stones, 0, -math.inf, math.inf, True)
            print(f"AI 剪枝了 {self.prune_times_in_each_turn} 次")

        print(f"AI 搜索了 {self.search_times_in_each_turn} 次")
        print(f"AI 得到的结果：{_}")

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
    STONE_NUM = 10
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
            algorithm = input("选择AI算法 (1: MinMax, 2: AlphaBeta): ").strip()
            algorithm = 'minmax' if algorithm == '1' else 'alphabeta'

            game.reset()
            player_turn = (choice == '1')  # True表示玩家先手

            print(f"\n游戏开始! {'玩家' if player_turn else 'AI'}先手")
            print(f"使用算法: {algorithm}")

            while game.get_current_stones() > 0:
                game.display_state()

                if player_turn:
                    # 玩家回合
                    while True:
                        try:
                            move = int(input(f"\n请取石子(1-{min(3, game.get_current_stones())}): "))
                            if game.player_move(move):
                                break
                            else:
                                print("无效的移动! 请重新输入。")
                        except ValueError:
                            print("请输入数字!")

                    if game.get_current_stones() == 0:
                        print("\n你取走了最后一颗石子，你赢了！")
                        break
                else:
                    # AI回合
                    print("\nAI思考中...")
                    move = game.get_ai_move(algorithm)
                    game.player_move(move)
                    print(f"AI取走了 {move} 颗石子")

                    if game.get_current_stones() == 0:
                        print("\nAI取走了最后一颗石子，AI赢了！")
                        break

                player_turn = not player_turn  # 切换回合

            # 询问是否再来一局
            play_again = input("\n是否再来一局? (y/n): ").strip().lower()
            if play_again != 'y':
                print("游戏结束！")
                break
