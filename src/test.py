from game import Game
from seller import SellerCurveType
from buyer import BuyerListGenerator, BuyerGeneratorType
from matplotlib import pyplot as plt

def test_game():
    N = 5
    m = 2
    # _t = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40]
    _t = [2, 4, 6, 8, 10, 12, 16, 20]
    s = 0
    p = 0
    regrets1 = []
    regrets2 = []
    times = 3
    
    buyer_gen = BuyerListGenerator(N, m)
    for t in _t:
        regret = 0
        for i in range(times):
            print("OK")
            game = Game(N, m, SellerCurveType.SMOOTH, buyer_generator=buyer_gen)
            sum, _regret, records, buyer_history = game.adversarial_online_pricing(t)
            regret += _regret
        regrets1.append(regret / times)
    
    with open("test_result.txt", "w") as f:
        f.write(str(regrets1))

    for t in _t:
        regret = 0
        for i in range(times):
            print("OK")
            game = Game(N, m, SellerCurveType.DIMINISHING, buyer_generator=buyer_gen)
            sum, _regret, records, buyer_history = game.adversarial_online_pricing(t)
            regret += _regret
        regrets2.append(regret / times)
    
    plt.plot(_t, regrets1, label="Smooth Curve")
    plt.scatter(_t, regrets1, marker='D')
    plt.plot(_t, regrets2, label="Diminishing Curve")
    plt.scatter(_t, regrets2, marker='D')
    plt.grid()
    plt.title("Adversarial Buyer Strategy")
    plt.xlabel("t")
    plt.ylabel("$R_t$")
    plt.legend()
    plt.savefig("test_result.png", dpi=300)

if __name__ == "__main__":
    test_game()
