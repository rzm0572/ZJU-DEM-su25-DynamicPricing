import numpy as np
import math
import itertools
from enum import Enum

from buyer import Buyer

class SellerCurveType(Enum):
    SMOOTH = 0
    DIMINISHING = 1

class SellerCurve:
    def __init__(self, N: int = 0, m: int = 0, jump_points: list = [], stage_values: list = []):
        self.N = N
        self.m = m
        self.jump_points = np.array(jump_points)
        self.stage_values = np.array(stage_values)
    
    def __str__(self):
        return f"SellerCurve(jump_points={self.jump_points}, stage_values={self.stage_values})"

    def get_stage(self, n: int):
        stage = np.searchsorted(self.jump_points, n, side='left')
        if stage >= self.m:
            stage = self.m - 1

        return stage
    
    def get_price(self, n: int) -> float:
        stage = self.get_stage(n)
        return self.stage_values[stage]


class SellerCurveGenerator:
    def __init__(self, N: int = 0):
        self.N = N
    
    def make_W(self, m, epsilon=0.1, begin=1):
        """
            m: 买家种类
            epsilon: 误差范围
        """

        # 构造等比数列 Z_i
        Z = [epsilon]
        while Z[-1] * (1 + epsilon) <= 1:
            Z.append(Z[-1] * (1 + epsilon))

        # 对 Z 的相邻元素作线性插值得到 W
        W = []
        for i in range(begin, len(Z)):
            for k in range(1, math.ceil((2+epsilon) * m) + 1):
                W.append(Z[i - 1] + Z[i - 1] * epsilon / m * k)
        return sorted(W)

    def discretize_smooth(self, N, m, epsilon, L):
        """
            N: 数据总量
            m: 买家种类
            epsilon: 误差范围
            L: 平滑系数
        """
        # W: 可选的价格集合
        W = self.make_W(m, epsilon)

        # delta: 离散化间隔
        delta = math.floor((epsilon * N) / (m * L))
        if delta == 0:
            delta = 1

        # N_S: 可取的跳跃点集合
        N_S = [delta * k for k in range(1, math.ceil(N / delta) + 1)]

        # 从 N_S 中任取 m 个元素作为跳跃点可以构造出一条 m-step 定价曲线
        # P: 所有满足条件的 m-step 定价曲线的集合
        N_combs  = list(itertools.combinations(N_S, m))
        W_combs = list(itertools.combinations(W, m))
        P = []
        for n_comb, w_comb in itertools.product(N_combs, W_combs):
            P.append(SellerCurve(N, m, list(n_comb), list(w_comb)))
        return P

    def discretize_diminishing(self, N, m, epsilon, J):
        """
            N: 数据总量
            m: 买家种类
            epsilon: 误差范围
            J: 边际收益递减常数
        """
        # W: 可选的价格集合
        W = self.make_W(m, epsilon, begin = 2)
        Y = [(2 * J * m) / (epsilon ** 2) * (1 + epsilon ** 2) ** i for i in range(math.ceil(math.log(N * epsilon ** 2 / (2 * J * m), 1 + epsilon ** 2)) + 1)]

        # N_D: 可取的跳跃点集合
        N_D = [k for k in range(1, math.ceil((2 * J * m) / (epsilon ** 2)) + 1)]
        print(len(Y))

        for i in range(len(Y) - 1):
            for k in range(math.floor(2 * J * m)):
                q_i = math.floor(Y[i] + Y[i] * (epsilon ** 2) / (2 * J * m) * k)
                print(i, k)
                if q_i not in N_D:
                    N_D.append(q_i)
        
        # 从 N_D 中任取 m 个元素作为跳跃点可以构造出一条 m-step 定价曲线
        # P: 所有满足条件的 m-step 定价曲线的集合
        N_combs = list(itertools.combinations(N_D, m))
        W_combs = list(itertools.combinations(W, m))
        P = []
        for n_comb, w_comb in itertools.product(N_combs, W_combs):
            P.append(SellerCurve(N, m, list(n_comb), list(w_comb)))
        return P



class Seller:
    def __init__(self, N: int = 0):
        self.N = N
        self.curve_generator = SellerCurveGenerator(N)
    
    def random_online_opt_curve(self, m: int, buyer: Buyer, Price_curves: list[SellerCurve], q: np.ndarray):
        max_rev = 0
        optimal_p = SellerCurve()
        for p in Price_curves:
            rev = 0
            for k in range(m):
                _, val = buyer.optimal_purchase(p)
                price = p.get_price(val)
                if val > 0:
                    rev += price * q[k]
            if rev > max_rev:
                max_rev = rev
                optimal_p = p
        
        return max_rev, optimal_p


