from train.lineWorldIa import LineWorldIa
import time


def launchLineWorld():

    lineWorld = LineWorldIa(11, "output/")

    start = time.time()
    values = lineWorld.iterativPolicyEvaluation(0.99, 0.00001)
    lineWorld.saveModel(values, "values_iter_pol_eval")
    print("Iterativ policy evaluation: " + str(time.time() - start))
    start = time.time()
    values, policies = lineWorld.policyIteration(0.99, 0.00001)
    lineWorld.saveModel(values, "values_pol_iter")
    lineWorld.saveModel(policies, "policies_pol_iter")
    print("Policy iteration: " + str(time.time() - start))
    start = time.time()
    values, policies = lineWorld.valueIteration(0.99, 0.00001)
    lineWorld.saveModel(values, "values_val_iter")
    lineWorld.saveModel(policies, "policies_val_iter")
    print("Value iteration: " + str(time.time() - start))
    start = time.time()
    values = lineWorld.firstVisitMonteCarlo(100, 100)
    lineWorld.saveModel(values, "values_frs_vis_monte_carlo")
    print("first visit montecarlo: " + str(time.time() - start))
    start = time.time()
    q, policies = lineWorld.monteCarloExplorControl(100, 100)
    lineWorld.saveModel(q, "probAct_monte_carlo_ec")
    lineWorld.saveModel(policies, "policies_monte_carlo_ec")
    print("Monte Carlo explorcontrol: " + str(time.time() - start))


launchLineWorld()