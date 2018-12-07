from PIL import Image, ImageDraw
from gurobipy import *
import numpy as np

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


def draw(listIm, indSecteurs, numVille, varNames, varVals):
    """crée la représentation graphique de la solution"""
    #une couleur par secteur
    colors = [(255,0,0),(0,255,0),(0,0,255), (128,0,0), (255,0,255), (255,255,0)]
    
    #repart de l'image initiale pour chaque solution
    im = Image.open("92.png")
    dr = ImageDraw.Draw(im)
    
    #trace les lignes du secteur vers la ville couverte
    for i in range(N):
        dr.line((affPos[i],affPos[numVille[i]]), fill=colors[indSecteurs[i]])
        
    #affiche les variables

    for i, (name, value) in enumerate(zip(varNames, varVals)):
        dr.text((100,500+15*i), str.ljust(name,5) + str(value), fill=0)
    
    #ajoute l'image à la liste pour la création du gif
    listIm.append(im)


def saveGif(listIm, name, kList, aList):
    #sauvegarde les images une par une
    cpt = 0
    for k in kList:
        for a in aList:
            listIm[cpt].save((name + "k" + str(k) + "_a" + str(a)), "PNG")
            cpt += 1
    
    #crée aussi un gif
    #récupère l'avant dernier caractère (numéro d'exercice) pour le nom du fichier
    listIm[0].save(name[-2] + ".gif", format="GIF", loop=9999,
                   duration=[4000] + [2000] * (len(listIm) - 2) + [4000],
                   save_all=True, append_images=listIm[1:])


def optimizeMean(pointsDacces, k:int, alpha:float, listIm=None):
    #crée le modèle
    m=Model("1"+str(k)+" "+str(alpha))
    #désactie les affichages
    m.Params.outPutFlag=0
    
    popMax = gamma(alpha, k)
    
    #extrait la matrice des distances des villes vers les points d'accès
    distSecteur = distVille[:, pointsDacces]
    
    lignes = range(N)
    colonnes = range(k)
    
    # declaration variables de decision
    x = np.empty((N, k), object)
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
        m.addConstr(quicksum(x[i][j] * pop[i] for i in lignes) <= popMax, "CPopMax%d" % j)
    
    # Résolution
    m.optimize()
    print(m)
    
    #secteur associé à chaque ville
    solSect = [max(colonnes, key=lambda _j: x[i][_j].X) for i in lignes]
    
    #ville associée à chaque ville
    solVille = pointsDacces[solSect]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax = max(sum(distSecteur[i][j] * x[i][j].X for j in range(k)) for i in lignes)
    
    if listIm is not None:
        #dessine
        draw(listIm, solSect, solVille, ["k", "a", "val", "dmax"],
             [k, alpha, round(val,1), dmax])
    
    print("k=", k, "a=", alpha, "\tvaleur:", round(val,3), "\tdmax:", dmax)
    
    return val

def optimizeMax(pointsDacces, k: int, alpha: float, f_optf: float, listeIm=None):
    """f_optf pour affichage du prix de l'équité"""
    m=Model("2"+str(k)+" "+str(alpha))
    m.Params.outPutFlag = 0
    
    popMax = gamma(alpha, k)
    
    #extrait la matrice des distances des villes vers les points d'accès
    distSecteur = distVille[:, pointsDacces]
    
    lignes = range(N)
    colonnes = range(k)
    
    # declaration variables de decision
    x = np.empty((N, k), object)
    maxDist = m.addVar(vtype=GRB.CONTINUOUS, name="maxDist")
    for i in lignes:
        for j in colonnes:
            x[i][j] = m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j))
    m.update()
    
    # definition de l'objectif
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distSecteur[i][j] * x[i][j]
    obj *= 1e-6
    
    obj += maxDist
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Definition des contraintes
    #une ville est dans un secteur exactement
    for i in lignes:
        m.addConstr(quicksum(x[i][j] for j in colonnes) == 1, "C1Secteur%d" % i)
    
    #un secteur ne couvre pas plus que gamma personnes
    for j in colonnes:
        m.addConstr(quicksum(x[i][j] * pop[i] for i in lignes) <= popMax, "CPopMax%d" % j)
    
    #maxDist>=dist
    for i in lignes:
        m.addConstr(maxDist >= quicksum(x[i][j] * distSecteur[i][j] for j in colonnes),
                    "maxDist%d" % i)
    
    # Résolution
    m.optimize()
    
    #secteur associé à chaque ville
    solSect = [max(colonnes, key=lambda _j: x[i][_j].X) for i in lignes]
    
    #ville associée à chaque ville
    solVille = pointsDacces[solSect]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax = maxDist.x
    
    f_optg = sum(distVille[i][solVille[i]] for i in lignes)
    prix_equite = round(100 - 100 * f_optf / f_optg, 1)
    
    if listeIm is not None:
        #dessine
        draw(listeIm, solSect, solVille, ["k", "a", "val", "dmax", "prix équité"],
             [k, alpha, round(val, 1), dmax, prix_equite])
    
    print("k=", k, "a=", alpha, "\tvaleur:", round(val, 3),
          "\tdmax:", dmax, "\tprix équité:", prix_equite)
    
    return val


def optimizeKMean(k: int, alpha: float, listIm=None):
    m = Model("m3, k=" + str(k) + " a=" + str(alpha))
    m.Params.outPutFlag = 0
    
    popMax = gamma(alpha, k)
    
    distSecteur = distVille
    
    lignes = range(N)
    colonnes = range(N)
    
    #-----VARIABLES-----
    #[i,j]=1 <=> i s'approvisionne en j
    whatSector = np.empty((N, N), object)
    for i in lignes:
        for j in colonnes:
            whatSector[i][j] = m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j))
    
    #isSector[j] <=> j est un secteur
    #             = max sur la colonne j
    isSector = np.empty((N,), object)
    for j in colonnes:
        isSector[j] = m.addVar(vtype=GRB.BINARY, name="VarIsSector%d" % j)
    
    m.update()
    
    #-----OBJECTIF-----
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distSecteur[i][j] * whatSector[i][j]
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    #------CONTRAINTES------
    #une ville est dans un secteur exactement
    for i in lignes:
        m.addConstr(quicksum(whatSector[i][j] for j in colonnes) == 1, "C1Secteur%d" % i)
    
    #un secteur est une ville approvisionnant au moins une ville
    for j in colonnes:
        for i in lignes:
            m.addConstr(isSector[j] >= whatSector[i][j], "CisSector%d%d" % (i, j))
    
    #un secteur ne couvre pas plus que gamma personnes
    #(rajoute aussi des contraintes inutiles disant qu'une ville
    #qui n'est pas un secteur doit couvrir moins que pop max)
    for j in colonnes:
        m.addConstr(quicksum(whatSector[i][j] * pop[i] for i in lignes) <= popMax, "CPopMax%d" % j)
    
    #il y a exactement k secteurs
    m.addConstr(quicksum(isSector[j] for j in colonnes) == k, "kSector")
    
    #----RESOLUTION-----
    m.optimize()
    
    #ville associée à chaque ville
    numVille = [max(colonnes, key=lambda _j: whatSector[i][_j].x) for i in lignes]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax = max(sum(distSecteur[i][j] * whatSector[i][j].X for j in colonnes) for i in lignes)
    
    villeToIndSecteur = {val: ind for ind, val, in enumerate(sorted(set(numVille)))}
    indSecteur = [villeToIndSecteur[v] for v in numVille]
    
    if listIm is not None:
        #dessine
        draw(listIm, indSecteur, numVille, ["k", "a", "val", "dmax"], [k, alpha, round(val, 1), dmax])
    
    print("k=", k, "a=", alpha, "\tvaleur:", round(val, 3), "\tdmax:", dmax)
    
    return val


def optimizeKMax(k, alpha, f_optf, listIm):
    m = Model("m3b, k=" + str(k) + " a=" + str(alpha))
    m.Params.outPutFlag = 0
    
    popMax = gamma(alpha, k)
    
    distSecteur = distVille
    
    lignes = range(N)
    colonnes = range(N)
    
    #-----VARIABLES-----
    #[i,j]=1 <=> i s'approvisionne en j
    whatSector = np.empty((N, N), object)
    for i in lignes:
        for j in colonnes:
            whatSector[i][j] = m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j))
    
    #isSector[j] <=> j est un secteur
    #             = max sur la colonne j
    isSector = np.empty((N,), object)
    for j in colonnes:
        isSector[j] = m.addVar(vtype=GRB.BINARY, name="VarIsSector%d" % j)
    
    #max dist
    maxDist = m.addVar(vtype=GRB.CONTINUOUS, name="maxDist")

    m.update()

    #-----OBJECTIF-----
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distSecteur[i][j] * whatSector[i][j]
    obj *= 1e-6

    obj += maxDist
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    #------CONTRAINTES------
    #maxDist>=dist
    for i in lignes:
        m.addConstr(maxDist >= quicksum(whatSector[i][j] * distSecteur[i][j] for j in colonnes),
                    "maxDist%d" % i)

    
    #une ville est dans un secteur exactement
    for i in lignes:
        m.addConstr(quicksum(whatSector[i][j] for j in colonnes) == 1, "C_1Secteur%d" % i)
    
    #un secteur est une ville approvisionnant au moins une ville
    for j in colonnes:
        for i in lignes:
            m.addConstr(isSector[j] >= whatSector[i][j], "C_isSector%d%d" % (i, j))
    
    #un secteur ne couvre pas plus que gamma personnes
    #(rajoute aussi des contraintes inutiles disant qu'une ville
    #qui n'est pas un secteur doit couvrir moins que pop max)
    for j in colonnes:
        m.addConstr(quicksum(whatSector[i][j] * pop[i] for i in lignes) <= popMax, "C_1PopMax%d" % j)
    
    #il y a exactement k secteurs
    m.addConstr(quicksum(isSector[j] for j in colonnes) == k, "kSector")
    
    #----RESOLUTION-----
    m.optimize()
    
    #ville associée à chaque ville
    numVille = [max(colonnes, key=lambda _j: whatSector[i][_j].x) for i in lignes]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax = maxDist.x
    
    villeToIndSecteur = {val: ind for ind, val, in enumerate(sorted(set(numVille)))}
    indSecteur = [villeToIndSecteur[v] for v in numVille]

    f_optg = sum(distVille[i][numVille[i]] for i in lignes)
    prix_equite = round(100 - 100 * f_optf / f_optg, 1)

    if listIm is not None:
        #dessine
        draw(listIm, indSecteur, numVille, ["k", "a", "val", "dmax", "prix équité"],
             [k, alpha, round(val, 1), dmax, prix_equite])
    
    print("k=", k, "a=", alpha, "\tvaleur:", round(val, 3), "\tdmax:", dmax)
    
    return val


def ex1(pointsDacces_k):
    #représentations graphiques des solutions
    images = []
    
    kList=[3,4,5]
    aList=[.1,.2]
    
    #nombre de secteurs
    for k in kList:
        #facteur de relaxation
        for a in aList:
            optimizeMean(pointsDacces_k[k], k, a, images)
    saveGif(images, "output/ex1/", kList, aList)

def ex2(pointsDacces_k):
    images = []
    kList=[3,4,5]
    aList=[.1,.2]
    for k in kList:
        for a in aList:
            v=optimizeMean(pointsDacces_k[k],k,a)
            optimizeMax(pointsDacces_k[k],k, a, v, images)
    saveGif(images, "output/ex2/", kList, aList)

def ex3():
    images = []
    kList=[3,4,5]
    aList=[.1,.2]
    vals=[]
    for k in kList:
        for a in aList:
            vals.append(optimizeKMean(k,a,images))
    saveGif(images, "output/ex3a/", kList, aList)
    images=[]
    for k in kList:
        for a in aList:
            optimizeKMax(k,a,vals.pop(0),images)
    saveGif(images, "output/ex3b/", kList, aList)


def main():
    #pour q1 et q2 on fixe quelles villes sont les points d'accès selon la valeur de k
    pointsDacces_k = [np.array(l) for l in
                      [[], [0], [0, 1], [1, 8, 14], [1, 13, 14, 22], [1, 13, 30, 32, 34]]]
    ex1(pointsDacces_k)
    ex2(pointsDacces_k)
    ex3()

main()
