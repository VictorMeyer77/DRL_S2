import pygame
from pygame.locals import *
from random import randint
from train.lineWorldIa import LineWorldIa
import time

positions = {0: (0, 250),
             1: (100, 250),
             2: (200, 250),
             3: (300, 250),
             4: (400, 250),
             5: (500, 250),
             6: (600, 250),
             7: (700, 250),
             8: (800, 250),
             9: (850, 250)}


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


pygame.init()

fenetre = pygame.display.set_mode((900, 700))
fond = pygame.image.load("media/img/lw_bg.jpg").convert_alpha()
fenetre.blit(fond, (0, 0))
pygame.display.flip()

line = pygame.image.load("media/img/line.png").convert_alpha()
line = pygame.transform.scale(line, (800, 100))
fenetre.blit(line, (50, 250))
pygame.display.flip()

troll = pygame.image.load("media/img/troll.png").convert_alpha()
troll = pygame.transform.scale(troll, (50, 50))
fenetre.blit(troll, positions[4])
pygame.display.flip()

troll2 = pygame.image.load("media/img/troll2.png").convert_alpha()
troll2 = pygame.transform.scale(troll2, (50, 50))
fenetre.blit(troll2, positions[4])
pygame.display.flip()

def updateMap():
    fenetre.blit(fond, (0, 0))
    fenetre.blit(line, (50, 250))
    fenetre.blit(troll2, positions[posIa])
    fenetre.blit(troll, positions[posRdm])
    pygame.display.flip()


run = True
posRdm = 4
posIa = 4

ia = LineWorldIa(11, "output/")
printChoiceModel()
model, modelType = chooseModel(ia)

while run:

    time.sleep(1)

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False

    posRdm = posRdm - 1 if randint(0, 1) == 0 else posRdm + 1
    posIa = posIa - 1 if getIaAction(model, modelType, posIa, ia) == 0 else posIa + 1

    updateMap()

    if posRdm < 1 or posIa == 9:
        printWin()

        if not continuer():
            run = False
        else:
            printChoiceModel()
            model, modelType = chooseModel(ia)
            posIa = 4
            posRdm = 4
            updateMap()

    elif posRdm == 9 or posIa < 1 or posIa == posRdm == 9:
        printLoose()
        if not continuer():
            run = False
        else:
            model, modelType = chooseModel(ia)
            posIa = 4
            posRdm = 4
            updateMap()
