import numpy as np
from game import Game, SellerCurveType

def main():
    game = Game(5, 2, SellerCurveType.SMOOTH)

    # sum, records, buyer_history = game.random_online_pricing(10)
    sum, records, buyer_history = game.adversarial_online_pricing(10)
    print(sum)
    for i, p in enumerate(records):
        print(f"buyer: {buyer_history[i]}, optimal_p: {p}")

if __name__ == "__main__":
    main()
