from PIL import Image, ImageDraw
from gurobipy import *
import numpy as np

#création du modèle gurobi
m = Model("affectation")

#désactive ses affichages
m.Params.outPutFlag = 0

#représentations graphiques des solutions
images = []


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

#lecture fichiers
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

def gamma(alpha, k):
    """nombre maximal de personnes couvertes par un secteur"""
    return (1+alpha)*popTot//k


def draw(solSect, solVille, varNames, varVals):
    """crée la représentation graphique de la solution"""
    #une couleur par secteur
    colors = [(255,0,0),(0,255,0),(0,0,255), (128,0,0), (255,0,255), (255,255,0)]
    
    #repart de l'image initiale pour chaque solution
    im = Image.open("92.png")
    dr = ImageDraw.Draw(im)
    
    #trace les lignes du secteur vers la ville couverte
    for i in range(N):
        dr.line((affPos[i],affPos[solVille[i]]), fill=colors[solSect[i]])
        
    #affiche les variables
    for i, (name, value) in enumerate(zip(varNames, varVals)):
        dr.text((100,500+15*i), str.ljust(name,5) + str(value), fill=0)
    
    #ajoute l'image à la liste pour la création du gif
    images.append(im)
    

def question1(k, alpha):
    popMax = gamma(alpha, k)
    
    #on fixe quelles villes sont les points d'accès (dépend de k)
    pointsDacces = np.array([[],[0],[0,1],[1,8,14],[1,13,14,22],[1,13,30,32,34]][k])
    
    #extrait la matrice des distances des villes vers les points d'accès
    distSecteur = distVille[:, pointsDacces]

    lignes = range(N)
    colonnes = range(k)
    
    # declaration variables de decision
    x = np.empty((N,k), object)
    for i in lignes:
        for j in colonnes:
            x[i][j] = m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j))
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
    
    # Résolution
    m.optimize()
    
    #secteur associé à chaque ville
    solSect = [max(colonnes, key=lambda j:x[i][j].x) for i in lignes]
    
    #ville associée à chaque ville
    solVille = pointsDacces[solSect]
    
    #valeur de l'objectif (tronquée)
    val = int(m.objVal)
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax= max(sum(distSecteur[i][j] * x[i][j].x for j in range(k)) for i in lignes)
    
    #dessine
    draw(solSect, solVille,["k","a","val","dmax"], [k,alpha,val,dmax])
    
    print("k=",k,"a=",alpha, "\tvaleur:", val, "\tdmax:", dmax)
    
    return val

def saveGif(name, kList, aList):
    #sauvegarde les images une par une
    cpt=0
    for k in kList:
        for a in aList:
            images[cpt].save((name+"k"+str(k)+"_a"+str(a)), "PNG")
            cpt+=1
    
    #crée aussi un gif
    images[0].save(name[-2]+".gif", format="GIF", loop=9999,
                   duration=[4000]+[2000]*(len(images)-2)+[4000],
                   save_all=True, append_images=images[1:])

def ex1():
    kList=[3,4,5]
    aList=[.1,.2]
    #nombre de secteurs
    for k in kList:
        #facteur de relaxation
        for a in aList:
            question1(k, a)
    saveGif("output/ex1/", kList, aList)
    
def ex2():
    kList=[3,4,5]
    aList=[.1,.2]
    #nombre de secteurs
    for k in kList:
        #facteur de relaxation
        for a in aList:
            question2(k, a)
    saveGif("output/ex2/", kList, aList)
    
    
def main():
    ex1()


main()