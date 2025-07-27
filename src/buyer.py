import numpy as np
from enum import Enum

class BuyerGeneratorType(Enum):
    RANDOM = 0
    ADVERSARIAL = 1

class Buyer:
    def __init__(self, N: int, values: np.ndarray | None = None):
        self.N = N
        if values is not None:
            self.values = values
        else:
            self.generate_values()

    def generate_values(self) -> np.ndarray:
        self.values = np.sort(
            np.concatenate([
                np.zeros(1),
                np.random.rand(self.N)
            ])
        )
        return self.values
    
    def optimal_purchase(self, P):
        """
            P: seller curves ,   shape: [m]
            V: value function,   shape: [N+1]
        """
        max_rev, val= 0, 0
        for i in range(1, len(self.values)):
            price = P.get_price(i)
            rev = self.values[i] - price
            if rev > max_rev:
                max_rev = rev
                val = i

        return max_rev, val
    
    def __str__(self) -> str:
        return f"Buyer with values {self.values}"

class BuyerListGenerator:
    def __init__(self, N: int, m: int):
        self.N = N
        self.m = m
        self.buyer_types = [Buyer(N) for _ in range(m)]
        self.history = []

        self.prob = np.random.rand(m)
        self.prob = self.prob / np.sum(self.prob)
    
    def get_buyer(self, i: int):
        return self.buyer_types[i]
        
    def choose_buyer_type(self, gen_type: BuyerGeneratorType = BuyerGeneratorType.RANDOM, optimal_p = None):
        buyer_type = 0
        if gen_type == BuyerGeneratorType.RANDOM:
            buyer_type = np.random.choice(range(self.m), p=self.prob)
        elif gen_type == BuyerGeneratorType.ADVERSARIAL:
            max_rev, buyer_type = 0, 0
            for i in range(self.m):
                rev, _ = self.buyer_types[i].optimal_purchase(optimal_p)
                if rev > max_rev:
                    max_rev = rev
                    buyer_type = i
        else:
            raise ValueError("Invalid generator type")

        self.history.append(buyer_type)
        return buyer_type, self.buyer_types[buyer_type]
    
    def expect_rev(self, optimal_p):
        rev = 0
        for i in range(self.m):
            _, val = self.buyer_types[i].optimal_purchase(optimal_p)
            price = optimal_p.get_price(val)
            rev += price * self.prob[i]
        return rev


if __name__ == '__main__':
    gen = BuyerListGenerator(4, 3)
    for i in range(10):
        buyer = gen.choose_buyer_type(BuyerGeneratorType.RANDOM)
        print(buyer)


