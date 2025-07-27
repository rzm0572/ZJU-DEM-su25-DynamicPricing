import numpy as np
import math
from buyer import Buyer, BuyerListGenerator
from seller import Seller, SellerCurve
from enum import Enum

class CurveType(Enum):
    SMOOTH = 0
    DIMINISHING = 1

class Game:
    def __init__(self, N: int, m: int, curve_type: CurveType = CurveType.SMOOTH, epsilon: float = 0.1, L: float = 10, J: float = 0.01):
        self.N = N
        self.m = m
        self.seller = Seller(N)

        self.P = []
        if curve_type == CurveType.SMOOTH:
            self.P = self.seller.curve_generator.discretize_smooth(N, m, epsilon, L)
        elif curve_type == CurveType.DIMINISHING:
            self.P = self.seller.curve_generator.discretize_diminishing(N, m, epsilon, J)
        else:
            raise ValueError("Invalid curve type")
    
    def choose_buyer_type(self, turn: int) -> int:
        return np.random.randint(0, self.m)
    
    def random_online_pricing(self, Time):
        gen = BuyerListGenerator(self.N, self.m, 0)

        T_bound = np.ones(self.m)
        T_fact = np.zeros(self.m)
        idx, _ = gen.choose_buyer_type()
        T_fact[idx] = 1

        sum = 0
        records = []

        for time in range(1, Time):
            q = np.zeros(self.m)
            for idx in range(self.m):
                q[idx] = T_fact[idx] / T_bound[idx] + math.sqrt(math.log(Time) / T_bound[idx])
            
            idx, buyer = gen.choose_buyer_type()
            max_rev, optimal_p = self.seller.random_online_opt_curve(self.m, buyer, self.P, q)

            sum += max_rev
            records.append(optimal_p)
            for k in range(self.m):
                _, val = buyer.optimal_purchase(optimal_p)
                if val > 0:
                    T_bound[k] += 1
                    if k == idx:
                        T_fact[k] += 1
        
        return sum, records, gen.history
    
    def adversarial_online_pricing(self, Time, theta=10):
        gen = BuyerListGenerator(self.N, self.m, 1)

        theta_p = [np.random.exponential(scale=1/theta) for _ in self.P]
        r_sum = np.zeros(len(self.P))
        sum = 0
        records = []
        for time in range(Time):
            optimal_p = self.P[np.argmax(r_sum + theta_p)]
            idx, buyer = gen.choose_buyer_type(optimal_p)
            _, val = buyer.optimal_purchase(optimal_p)

            for k, p in enumerate(self.P):
                if val > 0:
                    _, _val = buyer.optimal_purchase(p)
                    r_sum[k] += p.get_price(_val)
                else:
                    for j in range(self.m):
                        _, _val = gen.get_buyer(j).optimal_purchase(p)
                        if _val == 0:
                            r_sum[k] += p.get_price(_val)
            
            sum += optimal_p.get_price(val)
            records.append(optimal_p)
        return sum, records, gen.history


if __name__ == '__main__':
    game = Game(5, 2, CurveType.SMOOTH)

    sum, records, buyer_history = game.random_online_pricing(10)
    # sum, records, buyer_history = game.adversarial_online_pricing(10)
    print(sum)
    for i, p in enumerate(records):
        print(f"buyer: {buyer_history[i]}, optimal_p: {p}")
        
        
