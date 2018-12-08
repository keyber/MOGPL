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
            listIm[cpt].save(("output/"+name + "k" + str(k) + "_a" + str(a)), "PNG")
            cpt += 1
    
    #crée aussi un gif
    listIm[0].save(name + ".gif", format="GIF", loop=9999,
                   duration=[4000] + [2000] * (len(listIm) - 2) + [4000],
                   save_all=True, append_images=listIm[1:])


def _optimizeFixed(MINMAX, pointsDacces, k, alpha, listIm, f_optf=0):
    """Résout le problème de minimisation de la distance moyenne ou
    de la distance maximale (selon MINMAX)
    Les points d'accès sont fixés et donnés en paramètres
    k : nb points d'accès
    alpha : facteur de relaxation
    f_optf : valeur de l'optimum déjà obtenue pour MINMAX=False
            afin de calculer le coût de l'équité
    listIm : liste vide, ne crée pas d'images si mis à None
    """
    #crée le modèle
    m=Model("fixed k="+str(k)+" a="+str(alpha))
    #désactive les affichages
    m.Params.outPutFlag=0
    
    popMax = gamma(alpha, k)
    
    #extrait la matrice des distances des villes vers les points d'accès
    distSecteur = distVille[:, pointsDacces]
    
    lignes = range(N)
    colonnes = range(k)
    
    #-------------------VARIABLES-------------------
    x = np.empty((N, k), object)
    for i in lignes:
        for j in colonnes:
            x[i][j] = m.addVar(vtype=GRB.BINARY, name="ville%dsecteur%d" % (i, j))
    
    if MINMAX:
        maxDist = m.addVar(vtype=GRB.CONTINUOUS, name="maxDist")

    m.update()
    
    #-------------------OBJECTIF------------------
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distSecteur[i][j] * x[i][j]
    
    if MINMAX:
        obj *= 1e-4
        obj += maxDist
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    #------------------CONTRAINTES-------------------
    #une ville est dans un secteur exactement
    for i in lignes:
        m.addConstr(quicksum(x[i][j] for j in colonnes) == 1, "C1Secteur%d" % i)
    
    #un secteur ne couvre pas plus que gamma personnes
    for j in colonnes:
        m.addConstr(quicksum(x[i][j] * pop[i] for i in lignes) <= popMax, "CPopMax%d" % j)
    
    if MINMAX:
        #maxDist>=dist
        for i in lignes:
            m.addConstr(maxDist >= quicksum(x[i][j] * distSecteur[i][j] for j in colonnes),
                        "maxDist%d" % i)

    #------------------RESOLUTION------------------
    m.optimize()
    print("nb variables", len(m.getVars()), " contraintes", len(m.getConstrs()))
    
    #secteur associé à chaque ville
    solSect = [max(colonnes, key=lambda _j: x[i][_j].X) for i in lignes]
    
    #ville associée à chaque ville
    solVille = pointsDacces[solSect]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    #(c'est la valeur de maxDist.x si cette variable est définie)
    dmax = max(sum(distSecteur[i][j] * x[i][j].X for j in range(k)) for i in lignes)
    
    print("k=", k, "a=", alpha, "\tvaleur:", round(val,3), "\tdmax:", dmax)

    #-----------------PRIX EQUITE------------------
    if MINMAX:
        f_optg = sum(distVille[i][solVille[i]] for i in lignes)
        prix_equite = round(100 - 100 * f_optf / f_optg, 1)
        print("prix équité=",prix_equite)
        
    #-----------------DESSINE------------------
    if listIm is not None:
        nomVars = ["k", "a", "val", "dmax"]
        valVars = [k, alpha, round(val,4), dmax]
        if MINMAX:
            nomVars.append("prix équité")
            valVars.append(prix_equite)
        
        draw(listIm, solSect, solVille, nomVars,valVars)
 
    return val

def optimizeFixedMean(pointsDacces, k, alpha, listIm):
    return _optimizeFixed(0, pointsDacces, k, alpha, listIm)
def optimizeFixedMax(pointsDacces, k, alpha, listIm, f_optf):
    return _optimizeFixed(1, pointsDacces, k, alpha, listIm, f_optf)


def _optimizeFree(MINMAX, k, alpha, listIm, f_optf=0):
    """équivalent à optimizeFixed mais choisit ici les points d'accès de facon optimale"""
    m = Model("free, k=" + str(k) + " a=" + str(alpha))
    m.Params.outPutFlag = 0
    
    popMax = gamma(alpha, k)
    
    lignes = range(N)
    colonnes = range(N)
    
    #--------------------VARIABLES--------------------
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
    
    if MINMAX:
        maxDist = m.addVar(vtype=GRB.CONTINUOUS, name="maxDist")
    
    m.update()
    
    #--------------------OBJECTIF--------------------
    obj = LinExpr()
    for i in lignes:
        for j in colonnes:
            obj += distVille[i][j] * whatSector[i][j]
    
    if MINMAX:
        obj *= 1e-4
        obj += maxDist
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    #--------------------CONTRAINTES--------------------
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
    
    if MINMAX:
        #maxDist<=dist
        for i in lignes:
            m.addConstr(maxDist >= quicksum(whatSector[i][j] * distVille[i][j] for j in colonnes),
                        "maxDist%d" % i)

    #--------------------RESOLUTION--------------------
    m.optimize()
    
    #ville associée à chaque ville
    numVille = [max(colonnes, key=lambda _j: whatSector[i][_j].x) for i in lignes]
    
    #valeur de l'objectif
    val = m.objVal
    
    #distance maximale (valeur pour la ville la moins bien servie)
    dmax = max(sum(distVille[i][j] * whatSector[i][j].X for j in colonnes) for i in lignes)
    
    villeToIndSecteur = {val: ind for ind, val, in enumerate(sorted(set(numVille)))}
    indSecteur = [villeToIndSecteur[v] for v in numVille]

    #-----------------PRIX EQUITE------------------
    print("free" +("max" if MINMAX else "mean"), "k=", k, "a=", alpha, "\tvaleur:", val, "\tdmax:", dmax)
    print("nb variables", len(m.getVars()), " contraintes", len(m.getConstrs()))
    
    if MINMAX:
        f_optg = sum(distVille[i][numVille[i]] for i in lignes)
        prix_equite = round(100 - 100 * f_optf / f_optg, 1)
        print("prix équité=",prix_equite)

    if listIm is not None:
        #dessine
        nameVars = ["k", "a", "val", "dmax"]
        valVars = [k, alpha, round(val, 4), dmax]
        if MINMAX:
            nameVars.append("prix équité")
            valVars.append(prix_equite)
            
        draw(listIm, indSecteur, numVille, nameVars, valVars)
    
    return val


def optimizeFreeMean(k, alpha, listIm):
    return _optimizeFree(0, k, alpha, listIm)
def optimizeFreeMax(k, alpha, listIm, f_optf):
    return _optimizeFree(1, k, alpha, listIm, f_optf)


def ex12():
    #pour q1 et q2 on fixe quelles villes sont les points d'accès selon la valeur de k
    pointsDacces_k = [np.array(l) for l in
                      [[], [0], [0, 1], [1, 8, 14], [1, 13, 14, 22], [1, 13, 30, 32, 34]]]
    
    #listes des images pour créer un gif
    imagesMean = []
    imagesMax = []
    
    #paramètres qui varient
    kList=[3,4,5]
    aList=[.1,.2]
    
    for k in kList:
        for a in aList:
            #calcul de la moyenne minimale
            v=optimizeFixedMean(pointsDacces_k[k],k,a, imagesMean)
            
            #calcul du max minimal
            optimizeFixedMax(pointsDacces_k[k],k, a, imagesMax, v)
            print()
    
    #exporte les solutions trouvées
    saveGif(imagesMean, "fixed_mean", kList, aList)
    saveGif(imagesMax, "fixed_max", kList, aList)

def ex3():
    """comme ex 12 avec le choix des secteurs variable"""
    imagesMean = []
    imagesMax = []
    kList=[3,4,5]
    aList=[.1,.2]
    for k in kList:
        for a in aList:
            v = optimizeFreeMean(k, a, imagesMean)
            optimizeFreeMax(k, a, imagesMax, v)
            print()
    saveGif(imagesMean, "free_mean", kList, aList)
    saveGif(imagesMax, "free_max", kList, aList)


def main():
    ex12()
    ex3()

main()
