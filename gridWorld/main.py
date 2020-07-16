import pygame
import random
import time
from pygame.locals import *
from train.gridWorldIa import GridWorldIa

positions = {0: (80, 570),
             1: (300, 570),
             2: (520, 570),
             3: (740, 570),
             4: (80, 410),
             5: (300, 410),
             6: (520, 410),
             7: (740, 410),
             8: (80, 250),
             9: (300, 250),
             10: (520, 250),
             11: (740, 250),
             12: (80, 50),
             13: (300, 50),
             14: (520, 50),
             15: (740, 50)}


def printChoiceModel():
    print("Sélectionnez le modèle")
    print("a- Values Iterative Policy evaluation")
    print("b- Values policy iteration")
    print("c- Policies policy iteration")
    print("d- Values value iteration")
    print("e- Policies value iteration")
    print("f- Values first visit montecarlo")
    print("g- Policies montecarlo exploring control")
    print("h- ProbAct montecarlo exploring control")


def printLoose():
    print("Victoire Random")
    print("continuer ? Y/N")


def printWin():
    print("Victoire IA")
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


def chooseModel(ia):
    for event in pygame.event.get():

        if event.type == KEYDOWN:

            if event.key == K_a:
                return ia.loadModel("values_iter_pol_eval", "values"), "values"
            elif event.key == K_b:
                return ia.loadModel("values_pol_iter", "values"), "values"
            elif event.key == K_c:
                return ia.loadModel("policies_pol_iter", "policies"), "policies"
            elif event.key == K_d:
                return ia.loadModel("values_val_iter", "values"), "values"
            elif event.key == K_e:
                return ia.loadModel("policies_val_iter", "policies"), "policies"
            elif event.key == K_f:
                return ia.loadModel("values_frs_vis_monte_carlo", "values"), "values"
            elif event.key == K_g:
                return ia.loadModel("policies_monte_carlo_ec", "policies"), "policies"
            elif event.key == K_h:
                return ia.loadModel("probAct_monte_carlo_ec", "actProb"), "actProb"

    time.sleep(1)
    return chooseModel(ia)


def getIaAction(model, modelType, state, ia):
    assert modelType in ["values", "policies", "actProb"]

    if modelType == "values":
        return ia.getActionWithValues(model, state)
    elif modelType == "policies":
        return ia.getActionWithPolicies(model, state)
    elif modelType == "actProb":
        return ia.getActionWithActProb(model, state)


def getRdmAction(position):
    action = random.randint(0, 3)
    if action == 0:
        if position in [0, 4, 8, 12]:
            return getRdmAction(position)

    if action == 1:
        if position in [3, 7, 11, 15]:
            return getRdmAction(position)

    if action == 2:
        if position in [0, 1, 2, 3]:
            return getRdmAction(position)

    if action == 3:
        if position in [12, 13, 14, 15]:
            return getRdmAction(position)

    return action


def updateMap(posRdm, posIa):
    fenetre.blit(fond, (0, 0))
    fenetre.blit(troll2, positions[posIa])
    fenetre.blit(troll, positions[posRdm])
    pygame.display.flip()


pygame.init()

fenetre = pygame.display.set_mode((900, 700))
fond = pygame.image.load("media/img/bg_gw.png").convert()
fond = pygame.transform.scale(fond, (900, 700))
fenetre.blit(fond, (0, 0))
pygame.display.flip()

troll = pygame.image.load("media/img/troll.png").convert_alpha()
troll = pygame.transform.scale(troll, (80, 80))
fenetre.blit(troll, positions[0])
pygame.display.flip()

troll2 = pygame.image.load("media/img/troll2.png").convert_alpha()
troll2 = pygame.transform.scale(troll2, (80, 80))
fenetre.blit(troll2, positions[0])
pygame.display.flip()

run = True
posRdm = 0
posIa = 0

ia = GridWorldIa("output/")
printChoiceModel()
model, modelType = chooseModel(ia)

while run:

    time.sleep(1)

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False

    actionRdm = getRdmAction(posRdm)

    if actionRdm == 0:
        posRdm -= 1
    elif actionRdm == 1:
        posRdm += 1
    elif actionRdm == 2:
        posRdm -= 4
    elif actionRdm == 3:
        posRdm += 4

    actionIa = getIaAction(model, modelType, posIa, ia)

    if actionIa == 0:
        posIa -= 1
    elif actionIa == 1:
        posIa += 1
    elif actionIa == 2:
        posIa -= 4
    elif actionIa == 3:
        posIa += 4

    updateMap(posRdm, posIa)

    if posIa == 15:
        printWin()
        if not continuer():
            run = False
        else:
            printChoiceModel()
            model, modelType = chooseModel(ia)
            posIa = 0
            posRdm = 0
            updateMap(posRdm, posIa)

    elif posRdm == 15 or posIa == 3:
        printLoose()
        if not continuer():
            run = False
        else:
            printChoiceModel()
            model, modelType = chooseModel(ia)
            posIa = 0
            posRdm = 0
            updateMap(posRdm, posIa)
