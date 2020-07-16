import pandas as pd
import numpy as np

class GridWorldIa:

    def __init__(self, modelPath):

        self.modelPath = modelPath
        self.numStates = 16
        self.states = pd.DataFrame({"state": np.arange(self.numStates)})
        self.actions = pd.DataFrame({"action": [0, 1, 2, 3]})  # gauche droite bas haut
        self.terminal = pd.DataFrame({"state": [3, 15]})

        self.probs = {"state": [],
                      "action": [],
                      "nextState": [],
                      "reward": [],
                      "prob": []}
        self.setProbs()

    def setProbs(self):

        for state in self.states["state"]:

            for action in self.actions["action"]:

                for nextState in self.states["state"]:
                    self.initProb(state, action, nextState, 0, 0.0)
                    self.initProb(state, action, nextState, 1, 0.0)

        self.probs = pd.DataFrame(self.probs).set_index(["state", "action", "nextState", "reward"])
        for state in self.states["state"]:

            if state % 4 == 0:
                self.probs.loc[state, 0, state, 0] = 1.0
            else:
                self.probs.loc[state, 0, state - 1, 0] = 1.0
            if (state + 1) % 4 == 0:
                self.probs.loc[state, 1, state, 0] = 1.0
            else:
                self.probs.loc[state, 1, state + 1, 0] = 1.0
            if state < 4:
                self.probs.loc[state, 2, state, 0] = 1.0
            else:
                self.probs.loc[state, 2, state - 4, 0] = 1.0
            if state >= self.numStates - 4:
                self.probs.loc[state, 3, state, 0] = 1.0
            else:
                self.probs.loc[state, 3, state + 4, 0] = 1.0

        self.probs.loc[3, :, :, 0] = 0.0
        self.probs.loc[15, :, :, 0] = 0.0
        self.probs.loc[:, :, 3, 1] = -5.0
        self.probs.loc[:, :, 15, 1] = 1.0

    def initProb(self, state, action, nextState, reward, prob):
        self.probs["state"].append(state)
        self.probs["action"].append(action)
        self.probs["nextState"].append(nextState)
        self.probs["reward"].append(reward)
        self.probs["prob"].append(prob)

    def getPolicies(self):

        policies = {"state": [], "action": [], "prob": []}

        prob = 1.0 / self.actions.shape[0]

        for state in self.states["state"]:

            for action in self.actions["action"]:
                policies["state"].append(state)
                policies["action"].append(action)
                policies["prob"].append(prob)

        return pd.DataFrame(policies).set_index(["state", "action"])

    def initValueFunction(self):

        values = {"state": [], "esperance": []}

        for state in self.states["state"]:

            if state in self.terminal["state"].tolist():

                values["state"].append(state)
                values["esperance"].append(0.0)

            else:

                values["state"].append(state)
                values["esperance"].append(np.random.random())

        return pd.DataFrame(values).set_index(["state"])

    def getBestActionWithPolicies(self, policies, state):

        return policies.loc[state].idxmax()[0]

    def iterativPolicyEvaluation(self, gamma, theta, values=None, policies=None):

        if values is None:
            values = self.initValueFunction()
        if policies is None:
            policies = self.getPolicies()

        while True:

            delta = 0.0

            for state in self.states["state"]:

                value = values.at[state, "esperance"]
                total = 0.0

                for action in self.actions["action"]:

                    for nextState in self.states["state"]:
                        total += policies.loc[state].loc[action][0] * \
                                 self.probs.loc[state].loc[action].loc[nextState].loc[0][0] * \
                                 (self.probs.loc[state].loc[action].loc[nextState].loc[1][0] +
                                  gamma * values.loc[nextState][0])

                values.at[state, "esperance"] = total
                delta = np.maximum(delta, np.abs(total - value))

            if delta < theta:
                return values

    def policyIteration(self, gamma, theta):

        policies = self.getPolicies()
        values = self.initValueFunction()

        while True:

            values = self.iterativPolicyEvaluation(gamma, theta, values, policies)
            policyStable = True

            for state in self.states["state"]:

                oldAction = self.getBestActionWithPolicies(policies, state)
                bestAction = 0
                bestActionScore = -999

                for action in self.actions["action"]:

                    total = 0.0

                    for nextState in self.states["state"]:
                        total += self.probs.loc[state, action, nextState, 0][0] * \
                                 (self.probs.loc[state, action, nextState, 1][0] +
                                  gamma * values.loc[nextState][0])

                    if total > bestActionScore:
                        bestAction = action
                        bestActionScore = total

                policies.loc[state] = 0.0
                policies.loc[state, bestAction] = 1.0

                if bestAction != oldAction:
                    policyStable = False

            if policyStable:
                return values, policies

    def valueIteration(self, gamma, theta):

        values = self.initValueFunction()

        while True:

            delta = 0

            for state in self.states["state"]:

                value = values.loc[state][0]
                bestScore = -9999

                for action in self.actions["action"]:

                    total = 0.0

                    for nextState in self.states["state"]:
                        total += self.probs.loc[state, action, nextState, 0][0] * \
                                 (self.probs.loc[state, action, nextState, 1][0] +
                                  gamma * values.loc[nextState][0])

                    if total > bestScore:
                        bestScore = total

                values.loc[state] = bestScore
                delta = np.maximum(delta, np.abs(bestScore - value))

            if delta < theta:
                break

        policies = self.getPolicies()
        policies["prob"] = 0.0

        for state in self.states["state"]:

            bestAction = 0
            bestScore = -9999

            for action in self.actions["action"]:

                total = 0.0

                for nextState in self.states["state"]:
                    total += self.probs.loc[state, action, nextState, 0][0] * \
                             (self.probs.loc[state, action, nextState, 1][0] +
                              gamma * values.loc[nextState][0])

                if total > bestScore:
                    bestScore = total
                    bestAction = action

            policies.loc[state] = 0.0
            policies.loc[state, bestAction] = 1.0

        return values, policies

    def reset(self):
        return 0

    def isTerminal(self, state):
        return state in self.terminal["state"].tolist()

    def step(self, state, action):
        assert (state not in self.terminal["state"].tolist())
        nextState = np.random.choice(self.states["state"].tolist(),
                                     p=self.probs.loc[state, action, :, 0]["prob"].tolist())
        reward = self.probs.loc[state, action, nextState, 1]["prob"]
        return nextState, reward, (nextState in self.terminal["state"].tolist())

    def stepUntilEnd(self, state, policies, maxStep):

        history = {"state": [], "action": [], "nextState": [], "reward": []}
        curState = state
        stepCt = 0

        while not self.isTerminal(curState) and stepCt < maxStep:
            action = np.random.choice(self.actions["action"].tolist(), p=policies.loc[curState, :]["prob"].tolist())
            nextState, reward, isTerm = self.step(curState, action)
            history["state"].append(curState)
            history["action"].append(action)
            history["nextState"].append(nextState)
            history["reward"].append(reward)
            curState = nextState
            stepCt += 1
        return pd.DataFrame(history)

    def firstVisitMonteCarlo(self, nbEpisode=100000, maxStepPerEp=100, gamma=0.99, exploringStart=False):

        values = self.initValueFunction()
        policies = self.getPolicies()
        for state in self.states["state"]:
            if self.isTerminal(state):
                values.loc[state] = 0.0

        returns = np.zeros(values.shape[0])
        returnsCt = np.zeros(values.shape[0])

        for ep in range(nbEpisode):
            start = np.random.choice(self.states["state"].tolist()) if exploringStart else self.reset()
            history = self.stepUntilEnd(start, policies, maxStepPerEp)
            g = 0
            for i in reversed(range(history.shape[0])):
                g = gamma * g + history["reward"][i]
                curState = history["state"][i]
                if curState in history["state"].tolist()[0:i]:
                    continue
                returns[curState] += g
                returnsCt[curState] += 1
                values.loc[curState] = returns[curState] / returnsCt[curState]

        return values

    def monteCarloExplorControl(self, nbEpisode=10000, maxStepPerEp=100, gamma=0.99):

        policies = self.getPolicies()
        probByAct = pd.DataFrame({"state": self.states["state"],
                                  0: np.random.random(self.numStates),
                                  1: np.random.random(self.numStates),
                                  2: np.random.random(self.numStates),
                                  3: np.random.random(self.numStates)}).set_index("state")

        for state in self.states["state"]:
            if self.isTerminal(state):
                probByAct.loc[state] = 0.0
                policies.loc[state, :] = 0.0

        returns = np.zeros((self.states.shape[0], self.actions.shape[0]))
        returnsCt = np.zeros((self.states.shape[0], self.actions.shape[0]))

        for ep in range(nbEpisode):
            start = np.random.choice(self.states["state"].tolist())

            if self.isTerminal(start):
                continue

            firstAction = np.random.choice(self.actions["action"].tolist())
            nextState, reward, isTerm = self.step(start, firstAction)
            history = self.stepUntilEnd(nextState, policies, maxStepPerEp)

            statesList = [start] + history["state"].tolist()
            actionsList = [firstAction] + history["action"].tolist()
            rewardList = [reward] + history["reward"].tolist()

            g = 0
            for i in reversed(range(len(statesList))):

                g = gamma * g + rewardList[i]
                curState = statesList[i]
                curAction = actionsList[i]

                if curState in statesList[0:i] and curAction in actionsList[0:i]:
                    continue

                returns[curState, curAction] += g
                returnsCt[curState, curAction] += 1
                probByAct.loc[curState][curAction] = returns[curState, curAction] / returnsCt[curState, curAction]
                policies.loc[curState] = 0.0
                policies.loc[curState].loc[probByAct.loc[curState, :].idxmax()] = 1.0

        return probByAct, policies

    def loadModel(self, name, typeModel):

        assert typeModel in ["values", "policies", "actProb"]

        if typeModel == "values":
            return pd.read_csv(self.modelPath + name + ".csv").set_index("state")
        elif typeModel == "policies":
            return pd.read_csv(self.modelPath + name + ".csv").set_index(["state", "action"])
        elif typeModel == "actProb":
            return pd.read_csv(self.modelPath + name + ".csv").set_index("state")

    def saveModel(self, model, name):

        model.to_csv(self.modelPath + name + ".csv")

    def getActionWithValues(self, values, state):

        actions = {"action": [], "esperance": []}

        if state - 1 > 0:
            actions["action"].append(0)
            actions["esperance"].append(values.loc[state - 1]["esperance"])
        if state + 1 < self.numStates:
            actions["action"].append(1)
            actions["esperance"].append(values.loc[state + 1]["esperance"])
        if state - 4 > 0:
            actions["action"].append(2)
            actions["esperance"].append(values.loc[state - 4]["esperance"])
        if state + 4 < self.numStates:
            actions["action"].append(3)
            actions["esperance"].append(values.loc[state + 4]["esperance"])

        actions = pd.DataFrame(actions).set_index("action")
        return actions["esperance"].idxmax()

    def getActionWithPolicies(self, policies, state):
        return policies.loc[state].idxmax()[0]

    def getActionWithActProb(self, actProb, state):
        return int(actProb.loc[state].idxmax())

