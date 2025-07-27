import numpy as np
from game import Game, SellerCurveType
import click

@click.command()
@click.option("-n", default=5, help="Number of data")
@click.option("-m", default=2, help="Type of buyers")
@click.option("-t", default=10, help="Turns of the game")
@click.option("-s", default=0, help="Type of seller curve (0: smooth, 1: diminishing)")
@click.option("-p", default=0, help="Method of buyer generation (0: random, 1: adversarial)")
def main(n, m, t, s, p):
    if s == 0:
        curve_type = SellerCurveType.SMOOTH
    elif s == 1:
        curve_type = SellerCurveType.DIMINISHING
    else:
        raise ValueError("Invalid seller curve type")
    
    game = Game(n, m, curve_type)

    if p == 0:
        sum, regret, records, buyer_history = game.random_online_pricing(t)
    elif p == 1:
        sum, regret, records, buyer_history = game.adversarial_online_pricing(t)
    else:
        raise ValueError("Invalid buyer generation method")

    print(f"sum: {sum}")
    print(f"regret: {regret}")
    for i, p in enumerate(records):
        print(f"buyer: {buyer_history[i]}, optimal_p: {p}")

if __name__ == "__main__":
    main()
