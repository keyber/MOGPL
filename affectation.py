from PIL import Image, ImageDraw
from gurobipy import *
import numpy as np

N=36
#nom des villes
names = np.empty((N,), dtype=object)

#taille des villes
pop = np.empty((N,), dtype=int)

#matrice N*N des distances entre chaque couple ville ville
distVille = np.empty((N, N), dtype=float)

#position des villes pour l'affichage (en pixel)
affPos = np.empty((N, ), dtype=tuple)

with open("distances92.txt") as f:
    for i in range(N):
        names[i]=f.readline()[:-1]
        for j in range(N):
            distVille[i][j]=float(f.readline())
with open("populations92.txt") as f:
    for i in range(N):
        pop[i]=int(f.readline().split(',')[1])
with open("coordvilles92.txt") as f:
    for i in range(N):
        s = f.readline().split(',')[1:]
        affPos[i] = (int(s[0]), int(s[1]))

images=[]
im = Image.open("92.png")
draw = ImageDraw.Draw(im)
for i in range(N):
    draw.text(affPos[i],str(i),fill=0)
images.append(im)


#im.save("1", "PNG")
images[0].save("1", format="GIF", loop=9999, duration=1000, save_all=True, append_images=images[1:])



#population totale du territoire
popTot = pop.sum()

#nombre de secteurs
k = 3

#
alpha = 0.1

#
gamma = (1+alpha)*popTot/k

#
secteurs = [0,2,5]

#
distSecteur = distVille[:, secteurs]

#
chosenPoints = [[0,7,13],[0,12,13,21],[0,12,29,31,33]]



nbcont = 4
nbvar = 2

lignes = range(nbcont)
colonnes = range(nbvar)

# Matrice des contraintes
a = [[1, 0],
     [0, 1],
     [1, 2],
     [2, 1]]

# Second membre
b = [8, 6, 15, 18]

# Coefficients de la fonction objectif
c = [4, 10]

m = Model("mogplex")

# declaration variables de decision
x = []
for i in colonnes:
    x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="x%d" % (i + 1)))

# maj du modele pour integrer les nouvelles variables
m.update()

obj = LinExpr();
obj = 0
for j in colonnes:
    obj += c[j] * x[j]

# definition de l'objectif
m.setObjective(obj, GRB.MAXIMIZE)

# Definition des contraintes
for i in lignes:
    m.addConstr(quicksum(a[i][j] * x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)

# Resolution
m.optimize()

print("")
print('Solution optimale:')
for j in colonnes:
    print('x%d' % (j + 1), '=', x[j].x)
print("")
print('Valeur de la fonction objectif :', m.objVal)


