import math
import numpy as np
import itertools
from copy import deepcopy

def make_W(m, epsilon=0.1, begin=1):
    Z = [epsilon * (1 + epsilon) ** i for i in range(math.ceil(math.log(1 / epsilon, 1 + epsilon)) + 1)]
    W = []
    for i in range(begin, len(Z)):
        for k in range(1, math.ceil((2+epsilon) * m) + 1):
            W.append(Z[i - 1] + Z[i - 1] * epsilon / m * k)
    return sorted(W)

def discretize_smooth(N, m, epsilon, L):
    W = make_W(m, epsilon)
    delta = math.floor((epsilon * N) / (m * L))
    if delta == 0:
        delta = 1
    N_S = [delta * k for k in range(1, math.ceil(N / delta) + 1)]
    # print("Pricing Space : ", W)
    # print("Demand Space : ", N_S)
    N_combs  = list(itertools.combinations(N_S, m))
    W_combs = list(itertools.combinations(W, m))
    P = []
    for n_comb, w_comb in itertools.product(N_combs, W_combs):
        P.append([np.array(list(n_comb)), np.array(list(w_comb))])
    return P
    # record the m change points

def discretize_diminishing(N, m, epsilon, J):
    W = make_W(m, epsilon, begin = 2)
    Y = [(2 * J * m) / (epsilon ** 2) * (1 + epsilon ** 2) ** i for i in range(math.ceil(math.log(N * epsilon ** 2 / (2 * J * m), 1 + epsilon ** 2)) + 1)]
    N_D = [k for k in range(1, math.ceil((2 * J * m) / (epsilon ** 2)) + 1)]
    print(len(Y))
    for i in range(len(Y) - 1):
        for k in range(math.floor(2 * J * m)):
            q_i = math.floor(Y[i] + Y[i] * (epsilon ** 2) / (2 * J * m) * k)
            print(i, k)
            if q_i not in N_D:
                N_D.append(q_i)
    N_combs = list(itertools.combinations(N_D, m))
    W_combs = list(itertools.combinations(W, m))
    P = []
    for n_comb, w_comb in itertools.product(N_combs, W_combs):
        P.append([np.array(list(n_comb)), np.array(list(w_comb))])
    return P

def optimal_purchase(P, V):
    """P: [n_comb, w_comb] pairs,  shape: [m, 2]
       V: value function,          shape: [N+1]"""
    max_rev, val= 0, 0
    for i in range(1, len(V)):
        stage = np.searchsorted(P[0], i, side="left")
        if stage >= len(P[0]):
            stage = len(P[0]) - 1
        rev = V[i] - P[1][stage]
        if rev > max_rev:
            max_rev = rev
            val = i
    return max_rev, val
        
def random_online_pricing(m, types, Time, Value, Price_curves):
    T_bound = np.ones(m)
    T_fact = np.zeros(m)
    T_fact[types[0]] = 1
    sum = 0
    records = [0]
    for time in range(1, Time):
        q = np.zeros(m)
        for idx in range(m):
            q[idx] = T_fact[idx] / T_bound[idx] + math.sqrt(math.log(Time) / T_bound[idx])
        idx = types[time]
        # print(q)
        max_rev = 0
        optimal_p = None
        for p in Price_curves:
            rev = 0
            for k in range(m):
                _, val = optimal_purchase(p, Value[k])
                stage = np.searchsorted(p[0], val, side="left")
                if stage >= len(p[0]):
                    stage = len(p[0]) - 1
                if val > 0:
                    rev += p[1][stage] * q[k]
            if rev > max_rev:
                max_rev = rev
                optimal_p = p
        #     with open("records.txt", "a") as f:
        #         f.write(f"Price curve: {p}, Revenue: {rev}\n")
        # print(max_rev, optimal_p)
        a = optimal_purchase(optimal_p, Value[0])
        a = optimal_purchase(optimal_p, Value[1])
        sum += max_rev
        records.append(optimal_p)
        for k in range(m):
            _, val = optimal_purchase(optimal_p, Value[k])
            if val > 0:
                T_bound[k] += 1
                if k == idx:
                    T_fact[k] += 1
        
    return sum, records

def adversarial_online_pricing(m, types, Time, Value, Price_curves, theta=10):
    theta_p = [np.random.exponential(scale=1/theta) for _ in Price_curves]
    # r = np.zeros(len(Price_curves))
    # r_sum = deepcopy(r)
    r_sum = np.zeros(len(Price_curves))
    # print(theta_p)
    sum = 0
    records = []
    for time in range(Time):
        optimal_p = Price_curves[np.argmax(r_sum + theta_p)]
        idx = types[time]
        _, val = optimal_purchase(optimal_p, Value[idx])
        # f = open("records.txt", "a")

        for k, p in enumerate(Price_curves):
            stage = 0
            if val > 0:
                # print(p, Value[idx])
                stage = np.searchsorted(p[0], optimal_purchase(p, Value[idx])[1], side="left")
                if stage >= len(p[0]):
                    stage = len(p[0]) - 1
                r_sum[k] += p[1][stage]
            else:
                for j in range(m):
                    if optimal_purchase(p, Value[j])[1] == 0:
                        stage = np.searchsorted(p[0], optimal_purchase(p, Value[j])[1], side="left")
                        if stage >= len(p[0]):
                            stage = len(p[0]) - 1
                        r_sum[k] += p[1][stage]
            # f.write(f"Time: {time}, Optimal Price Curve: {p}, Value: {val}, Stage: {stage}, Revenue: {r_sum[k]}\n")    
        # print(val)
        # print(optimal_p)
        stage = np.searchsorted(optimal_p[0], val, side="left")
        if stage >= len(optimal_p[0]):
            stage = len(optimal_p[0]) - 1
        sum += optimal_p[1][stage]
        records.append(optimal_p)
    return sum, records

def main():
    N = 5
    m = 2
    P = discretize_smooth(N, m, epsilon=0.1, L=10)
    # P = discretize_diminishing(N, m, epsilon=0.1, J=0.01)
    value_curve = []
    value_curve.append([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    value_curve.append([0, 0.4, 0.7, 0.9, 0.95, 0.97])
    # Time = 5
    Time = 10
    # i = [0, 0, 1, 0, 1]
    types = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # sum, records = random_online_pricing(m, types, Time, value_curve, P)
    sum, records = adversarial_online_pricing(m, types, Time, value_curve, P, theta=10)
    print(sum)
    print(records)
    # W = make_W(m, epsilon)

if __name__ == "__main__":
    main()