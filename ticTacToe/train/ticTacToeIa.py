import pandas as pd
import numpy as np
import random

class TicTacToeIa:

    def __init__(self, modelPath):

        self.modelPath = modelPath
        self.winCombs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]
        self.actions = np.arange(1, 10)
        self.states = self.getStates()
        self.terminal = self.getTerminal()

    def getStates(self):

        tourDict = {0: self.getInitStates(), 1: None, 2: None, 3: None, 4: None, 5: None}

        for i in range(1, 6):

            nextStates = []
            for state in tourDict[i - 1]:
                if self.isWinComb(state) or self.isLooseComb(state):
                    continue

                for action in self.actions:
                    if state[action - 1] != 0:
                        continue

                    tmpNextStates = self.getNextStates(state, action)
                    if len(tmpNextStates) > 0:
                        nextStates.append(self.getNextStates(state, action))

            tourDict[i] = np.concatenate(nextStates, axis=0)

        return np.unique(np.concatenate([tour for tour in tourDict.values()], axis=0), axis=0)

    def getInitStates(self):

        initStates = []
        init = np.array([0] * 9)
        initStates.append(init)

        for action in self.actions:
            nextState = init.copy()
            nextState[action - 1] = 2
            initStates.append(nextState)

        return np.array(initStates)

    def getNextStates(self, state, action):

        assert state[action - 1] == 0

        nextStates = []
        newState = state.copy()
        newState[action - 1] = 1

        if self.isWinComb(newState) or len(np.where(newState == 0)[0]) < 1:
            return newState.reshape((1, 9))

        for ennemyAction in self.actions:

            if newState[ennemyAction - 1] != 0:
                continue

            else:
                ennemyState = newState.copy()
                ennemyState[ennemyAction - 1] = 2
                nextStates.append(ennemyState)

        return np.array(nextStates)

    def getPolicies(self):

        policies = np.zeros((len(self.states), len(self.actions)))

        for i in range(len(self.states)):
            if self.isTerminal(self.states[i]):
                continue
            indexPossible = np.where(self.states[i] == 0)
            policies[i, indexPossible] = 1.0 / len(indexPossible[0])

        return policies

    def initValues(self):
        values = np.zeros((len(self.states)))
        for i in range(len(values)):
            if not self.isTerminal(self.states[i]):
                values[i] = np.random.random()
        return values

    def getTerminal(self):

        term = []

        for state in self.states:

            if self.isWinComb(state) or self.isLooseComb(state) or len(np.where(state == 0)[0]) < 1:
                term.append(state)
                continue

        return np.array(term)

    def isWinComb(self, state):

        for wc in self.winCombs:

            isWin = True
            for case in wc:
                if state[case - 1] != 1:
                    isWin = False
            if isWin:
                return True

        return False

    def isLooseComb(self, state):

        for wc in self.winCombs:

            isLoose = True
            for case in wc:
                if state[case - 1] != 2:
                    isLoose = False

            if isLoose:
                return True

        return False

    def getStatesIndex(self, state):

        for i in range(len(self.states)):
            if str(state) == str(self.states[i]):
                return i
        return -1

    def getRandomNextState(self, state, action):

        assert not self.isTerminal(state)
        nextStates = self.getNextStates(state, action)
        stateIndex = random.randint(0, len(nextStates) - 1) if len(nextStates) > 1 else 0
        rdmState = nextStates[stateIndex]
        probWin = 1 if self.isWinComb(rdmState) else 0
        return rdmState, probWin

    def reset(self):
        initStates = self.getInitStates()
        return initStates[random.randint(0, len(initStates) - 1)]

    def isTerminal(self, state):
        for term in self.terminal:
            if np.array_equal(state, term):
                return True
        return False

    def step(self, state, action):

        assert not self.isTerminal(state)
        nextState, reward = self.getRandomNextState(state, action)
        return nextState, reward, self.isTerminal(nextState)

    def stepUntilEnd(self, state, policies, maxStep):

        states = []
        actions = []
        nextStates = []
        rewards = []
        curState = state.copy()
        stepCt = 0

        while not self.isTerminal(curState) and stepCt < maxStep:
            action = np.random.choice(self.actions, p=policies[self.getStatesIndex(curState)])
            nextState, reward, isTerm = self.step(curState, action)
            states.append(curState)
            actions.append(action)
            nextStates.append(nextState)
            rewards.append(reward)
            curState = nextState.copy()
            stepCt += 1
        return states, actions, nextStates, rewards

    def monteCarloExplorControl(self, nbEpisode=1000, maxStepPerEp=5, gamma=0.99):

        policies = self.getPolicies()

        probByAct = {"state": range(len(self.states)),
                     1: np.random.random(len(self.states)),
                     2: np.random.random(len(self.states)),
                     3: np.random.random(len(self.states)),
                     4: np.random.random(len(self.states)),
                     5: np.random.random(len(self.states)),
                     6: np.random.random(len(self.states)),
                     7: np.random.random(len(self.states)),
                     8: np.random.random(len(self.states)),
                     9: np.random.random(len(self.states))}

        probByAct = pd.DataFrame(probByAct).set_index("state")

        for i in range(len(self.states)):
            if self.isTerminal(self.states[i]):
                probByAct.loc[i, :] = 0.0

        returns = np.zeros((len(self.states), len(self.actions)))
        returnsCt = np.zeros((len(self.states), len(self.actions)))

        for ep in range(nbEpisode):

            start = self.states[random.randint(0, len(self.states) - 1)]

            if self.isTerminal(start):
                continue

            firstAction = np.random.choice(np.where(start == 0)[0]) + 1
            nextState, reward, isTerm = self.step(start, firstAction)
            states, actions, nextStates, rewards = self.stepUntilEnd(nextState, policies, maxStepPerEp)

            states = [start] + states
            actions = [firstAction] + actions
            rewards = [reward] + rewards

            g = 0
            for i in reversed(range(len(states))):

                g = gamma * g + rewards[i]
                curState = states[i].copy()
                curAction = actions[i]

                if curAction in actions[0:i]:
                    isAlreadyUse = False
                    for oldState in states[0:i]:
                        if np.array_equal(oldState, curState):
                            isAlreadyUse = True
                            break
                    if isAlreadyUse:
                        continue

                returns[i, curAction - 1] += g
                returnsCt[i, curAction - 1] += 1
                probByAct.loc[i][curAction - 1] = returns[i, curAction - 1] / returnsCt[i, curAction - 1]
                policies[i] = 0.0
                policies[i][probByAct.loc[i, :].idxmax() - 1] = 1.0

        return probByAct, policies

    def loadModel(self, name, typeModel):

        assert typeModel in ["values", "policies", "actProb"]

        if typeModel == "values":
            return pd.read_csv(self.modelPath + name + ".csv").set_index("state")
        elif typeModel == "policies":
            return pd.read_csv(self.modelPath + name + ".csv").set_index("state")
        elif typeModel == "actProb":
            return pd.read_csv(self.modelPath + name + ".csv").set_index("state")

    def saveModel(self, model, name):

        model.to_csv(self.modelPath + name + ".csv")

    def savePolicies(self, policies, name):
        df = {"state": range(len(policies)), "action": policies.argmax(axis=1) + 1}
        df = pd.DataFrame(df).set_index("state")
        df.to_csv(self.modelPath + name + ".csv")

    def getActionWithActProb(self, actProb, state):
        return int(actProb.loc[self.getStatesIndex(state)].idxmax())

    def getActionWithPolicies(self, policies, state):
        return int(policies.at[self.getStatesIndex(state), "action"])