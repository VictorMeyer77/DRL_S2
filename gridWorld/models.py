from train.gridWorldIa import GridWorldIa
import time

def laucnhGridWorld():

    gridWorld = GridWorldIa("output/")

    start = time.time()
    values = gridWorld.iterativPolicyEvaluation(0.99, 0.00001)
    gridWorld.saveModel(values, "values_iter_pol_eval")
    print("Iterativ policy evaluation: " + str(time.time() - start))
    start = time.time()
    values, policies = gridWorld.policyIteration(0.99, 0.00001)
    gridWorld.saveModel(values, "values_pol_iter")
    gridWorld.saveModel(policies, "policies_pol_iter")
    print("Policy iteration: " + str(time.time() - start))
    start = time.time()
    values, policies = gridWorld.valueIteration(0.99, 0.00001)
    gridWorld.saveModel(values, "values_val_iter")
    gridWorld.saveModel(policies, "policies_val_iter")
    print("Value iteration: " + str(time.time() - start))
    start = time.time()
    values = gridWorld.firstVisitMonteCarlo(10000, 100)
    gridWorld.saveModel(values, "values_frs_vis_monte_carlo")
    print("first visit montecarlo: " + str(time.time() - start))
    start = time.time()
    q, policies = gridWorld.monteCarloExplorControl(10000, 100)
    gridWorld.saveModel(q, "probAct_monte_carlo_ec")
    gridWorld.saveModel(policies, "policies_monte_carlo_ec")
    print("Monte Carlo explorcontrol: " + str(time.time() - start))


laucnhGridWorld()