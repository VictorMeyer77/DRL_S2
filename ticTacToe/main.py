import pygame
import time
from pygame.locals import *
import numpy as np
from train.ticTacToeIa import TicTacToeIa


POSITIONS = {1: (100, 50),
             2: (360, 50),
             3: (620, 50),
             4: (100, 260),
             5: (360, 260),
             6: (620, 260),
             7: (100, 470),
             8: (360, 470),
             9: (620, 470)}

INIT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

def displayPlayer(player, position):

    assert player in [1, 2] and position in POSITIONS.keys()

    if player == 1:
        fenetre.blit(troll2, POSITIONS[position])
    else:
        fenetre.blit(troll1, POSITIONS[position])

    pygame.display.flip()

def choosePlayerType():

    for event in pygame.event.get():

        if event.type == KEYDOWN:

            if event.key == K_a:
                return "human"
            elif event.key == K_b:
                return "ia"
            elif event.key == K_c:
                return "random"

    time.sleep(1)
    return choosePlayerType()

def startGame():
    print("a- humain")
    print("b- ia")
    print("c- random")
    print("Choisissez le type du joueur 1: ")
    playerOne = choosePlayerType()
    print("Choisissez le type du joueur 2: ")
    playerTwo = choosePlayerType()

    return playerOne, playerTwo

def printEndGame(end):

    assert end in [1, 2, 3]

    if end == 1:
        print("Victoire Joueur 1")
    elif end == 2:
        print("Victoire Joueur 2")
    else:
        print("Egalité")
    print("continuer ? Y/N")


def continuer():

    for event in pygame.event.get():

        if event.type == KEYDOWN:

            if event.key == K_y:
                return True
            if event.key == K_n:
                return False

    time.sleep(1)
    return continuer()

def printChooseModel():
    print("Sélectionnez le modèle")
    print("a- probAct montecarlo with exploring start control")
    print("b- policies montecarlo with exploring start control")

def chooseModel(ia):

    for event in pygame.event.get():

        if event.type == KEYDOWN:

            if event.key == K_a:
                return ia.loadModel("probAct_monte_carlo_ec", "actProb"), "actProb"
            elif event.key == K_b:
                return ia.loadModel("policies_monte_carlo_ec", "policies"), "policies"

    time.sleep(1)
    return chooseModel(ia)

def getRandomAction(state):
    return np.random.choice(np.where(state == 0)[0]) + 1

def getHumanAction(state):

    action = -1
    for event in pygame.event.get():

        if event.type == KEYDOWN:

            if event.key == K_a:
                action = 1
            elif event.key == K_b:
                action = 2
            elif event.key == K_c:
                action = 3
            elif event.key == K_d:
                action = 4
            elif event.key == K_e:
                action = 5
            elif event.key == K_f:
                action = 6
            elif event.key == K_g:
                action = 7
            elif event.key == K_h:
                action = 8
            elif event.key == K_i:
                action = 9

    if action - 1 in np.where(state == 0)[0]:
        return action
    else:
        time.sleep(1)
        return getHumanAction(state)

def inverseState(state):

    inverseState = state.copy()
    for i in range(len(inverseState)):
        if inverseState[i] == 1:
            inverseState[i] = 2
        elif inverseState[i] == 2:
            inverseState[i] = 1
    return inverseState

def getIaAction(state, model, modelType, ia):

    assert modelType in ["actProb", "policies"]

    if modelType == "actProb":
        return ia.getActionWithActProb(model, state)
    if modelType == "policies":
        return ia.getActionWithPolicies(model, state)


pygame.init()

fenetre = pygame.display.set_mode((900, 700))
fond = pygame.image.load("media/img/background.png").convert()
fond = pygame.transform.scale(fond, (900, 700))
fenetre.blit(fond, (0, 0))
pygame.display.flip()

troll1 = pygame.image.load("media/img/troll.png").convert_alpha()
troll1 = pygame.transform.scale(troll1, (150, 150))
troll2 = pygame.image.load("media/img/troll2.png").convert_alpha()
troll2 = pygame.transform.scale(troll2, (150, 150))


print("Chargement ia...")
ia = TicTacToeIa("output/")
run = True
modelOne, modelTypeOne = None, None
modelTwo, modelTypeTwo = None, None
playerOneType, playerTwoType = startGame()


if playerOneType == "ia":
    print("Joueur 1")
    printChooseModel()
    modelOne, modelTypeOne = chooseModel(ia)

if playerTwoType == "ia":
    print("Joueur 2")
    printChooseModel()
    modelTwo, modelTypeTwo = chooseModel(ia)


state = INIT.copy()

while run:

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False

    action = -1
    if playerOneType == "random":
        action = getRandomAction(state)
    elif playerOneType == "human":
        action = getHumanAction(state)
    elif playerOneType == "ia":
        action = getIaAction(state, modelOne, modelTypeOne, ia)

    state[action - 1] = 1
    displayPlayer(1, action)
    time.sleep(1)

    if ia.isWinComb(state):
        printEndGame(1)
        if not continuer():
            run = False
            continue
        else:
            playerOneType, playerTwoType = startGame()
            if playerOneType == "ia":
                print("Joueur 1")
                printChooseModel()
                modelOne, modelTypeOne = chooseModel(ia)

            if playerTwoType == "ia":
                print("Joueur 2")
                printChooseModel()
                modelTwo, modelTypeTwo = chooseModel(ia)
            state = INIT.copy()
            fenetre.blit(fond, (0, 0))
            pygame.display.flip()

    elif len(np.where(state == 0)[0]) < 1:
        printEndGame(3)
        if not continuer():
            run = False
            continue
        else:
            playerOneType, playerTwoType = startGame()
            if playerOneType == "ia":
                print("Joueur 1")
                printChooseModel()
                modelOne, modelTypeOne = chooseModel(ia)

            if playerTwoType == "ia":
                print("Joueur 2")
                printChooseModel()
                modelTwo, modelTypeTwo = chooseModel(ia)
            state = INIT.copy()
            fenetre.blit(fond, (0, 0))
            pygame.display.flip()

    action = -1
    if playerTwoType == "random":
        action = getRandomAction(state)
    elif playerTwoType == "human":
        action = getHumanAction(state)
    elif playerTwoType == "ia":
        action = getIaAction(inverseState(state), modelTwo, modelTypeTwo, ia)

    state[action - 1] = 2
    displayPlayer(2, action)
    time.sleep(1)

    if ia.isLooseComb(state):
        printEndGame(2)
        if not continuer():
            run = False
            continue
        else:
            playerOneType, playerTwoType = startGame()
            if playerOneType == "ia":
                print("Joueur 1")
                printChooseModel()
                modelOne, modelTypeOne = chooseModel(ia)

            if playerTwoType == "ia":
                print("Joueur 2")
                printChooseModel()
                modelTwo, modelTypeTwo = chooseModel(ia)
            state = INIT.copy()
            fenetre.blit(fond, (0, 0))
            pygame.display.flip()

    elif len(np.where(state == 0)[0]) < 1:
        printEndGame(3)
        if not continuer():
            run = False
            continue
        else:
            playerOneType, playerTwoType = startGame()
            if playerOneType == "ia":
                print("Joueur 1")
                printChooseModel()
                modelOne, modelTypeOne = chooseModel(ia)

            if playerTwoType == "ia":
                print("Joueur 2")
                printChooseModel()
                modelTwo, modelTypeTwo = chooseModel(ia)
            state = INIT.copy()
            fenetre.blit(fond, (0, 0))
            pygame.display.flip()
