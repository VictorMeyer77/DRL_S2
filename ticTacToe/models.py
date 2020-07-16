from train.ticTacToeIa import TicTacToeIa
import time

def laucnhTicTacToe():

    ticTacToe = TicTacToeIa("output/")

    start = time.time()
    q, policies = ticTacToe.monteCarloExplorControl(100000, 5)
    ticTacToe.saveModel(q, "probAct_monte_carlo_ec")
    ticTacToe.savePolicies(policies, "policies_monte_carlo_ec")
    print("Monte Carlo explorcontrol: " + str(time.time() - start))

laucnhTicTacToe()