from PIL import Image, ImageDraw
from gurobipy import *
import numpy as np

#création du modèle gurobi
m = Model("affectation")

#désactive ses affichages
m.Params.outPutFlag = 0


#nombre de ville
N=36

#nom des villes
names = np.empty((N,), dtype=object)

#taille des villes
pop = np.empty((N,), dtype=int)

#matrice N*N des distances entre chaque ville
distVille = np.empty((N, N), dtype=float)

#position des villes (pour l'affichage) (en pixel)
affPos = np.empty((N, ), dtype=tuple)

with open("distances92.txt") as f:
    for _i in range(N):
        names[_i]= f.readline()[:-1]
        for _j in range(N):
            distVille[_i][_j]=float(f.readline())
with open("populations92.txt") as f:
    for _i in range(N):
        pop[_i]=int(f.readline().split(',')[1])
with open("coordvilles92.txt") as f:
    for _i in range(N):
        s = f.readline().split(',')[1:]
        affPos[_i] = (int(s[0]), int(s[1]))

#population totale du territoire
popTot = pop.sum()

#nombre de secteurs
k = 3

#facteur de relaxation
alpha = 0.1

#gamma
popMax = (1+alpha)*popTot//k



def draw():
    images=[]
    im = Image.open("92.png")
    dr = ImageDraw.Draw(im)
    for i in range(N):
        dr.text(affPos[i],str(i),fill=0)
    images.append(im)
    
    
    #im.save("1", "PNG")
    images[0].save("1", format="GIF", loop=9999, duration=1000,
                   save_all=True, append_images=images[1:])

def question1():
    #on fixe quelles villes sont les points d'accès (dépend de k)
    chosenPoints = [[],[0],[0,1],[0,7,13],[0,12,13,21],[0,12,29,31,33]][k]
    
    #extrait la matrice des distances des villes vers les points d'accès
    distSecteur = distVille[:, chosenPoints]

    lignes = range(N)
    colonnes = range(k)
    
    # declaration variables de decision
    x = []
    for i in lignes:
        x.append([])
        for j in colonnes:
            x[i].append(m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j)))
    m.update()
    
    # definition de l'objectif
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distSecteur[i][j] * x[i][j]
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Definition des contraintes
    #une ville est dans un secteur exactement
    for i in lignes:
        m.addConstr(quicksum(x[i][j] for j in colonnes) == 1, "C1Secteur%d" % i)
    
    #un secteur ne couvre pas plus que gamma personnes
    for j in colonnes:
        m.addConstr(quicksum(x[i][j]*pop[i] for i in lignes) <= popMax, "CPopMax%d" % j)
    
    # Resolution
    m.optimize()
    
    print("")
    print('Solution optimale:')
    for i in lignes:
        print("ville",i,"secteur",max(colonnes, key=lambda j:x[i][j].x))
    print("")
    print('Valeur de la fonction objectif :', m.objVal)


question1()