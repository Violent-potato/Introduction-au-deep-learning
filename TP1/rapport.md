# Ex 1: Utilisation de SLURM (∼30mn, – facile)
## c
Avant d'allouer des GPU:
![1.c_before.png](images/1.c_before.png)

je m'étais trompé.e, il ne fallait pas aller sur le LAB_GATEWAY mais rester sur le controlleur. Le message d'erreur sur le contrôlleur était similaire.
Après avoir exécuté ___nvidia-smi___ on obtient:
![1.c_during.png](images/1.c_during.png)
![1.c_after.png](images/1.c_after.png)

Le modèle de GPU NVIDIA L4 m'a été alloué.
## d
![1.d.png](images/1.d.png)

___scancel 1718___
## e
![1.e.png](images/1.e.png)

Le nom exact du fichier est ___hello-slurm-1734.out___
## f
![1.f.png](images/1.f.png)

___ReqMem___ correspond à "requested memory", soit la mémoire allouée pour cette tâche. On y retrouve en effet les 8G de RAM alloués précédemment.

___MaxRSS___, lui, correspond à la mémoire utilisée pour réaliser la tâche.

# Ex 2: Création d'un environnement virtuel Python (∼20mn, – facile)
## b
La commande pour obtenir la version est ___python --version___. Pour le chemin du binaire, c'est ___which python___
![2.b.png](images/2.b.png)

## d
## e
## f
# Ex 3: Exercices théoriques (Papier & Markdown) (∼30mn, – moyen) (à faire en dehors de TSP)
## a
![3.a.jpg](images/3.a.jpg)

Sans le biais:

$$ Y = X W_1^T $$

$$ Z = Y W_2^T $$

Soient $X \in \mathbb{R}^{1 \times 3}$, $Y \in \mathbb{R}^{1 \times 4}$ et $Z \in \mathbb{R}^{1 \times 2}$.

On a donc $W_1 \in \mathbb{R}^{4 \times 3}$ et $W_2 \in \mathbb{R}^{2 \times 4}$

*Couche 1 et cachée*: $4 \times 3$ = 12 poids

*Couche cachée et de sortie*: $4 \times 2$ = 8 poids

Il y a **20** paramètres sans biais.

Pour la couche cachée, le biais est une matrice qui appartient à $\mathbb{R}^{1 \times 4}$

Pour la couche cachée, le biais est une matrice qui appartient à $\mathbb{R}^{1 \times 2}$

Soient **26** paramètres avec biais.

## b
H = ReLU( X · W1^T + b1 )
Y = H · W2^T + b2

Soient m1, m2 dans R
Dimensions :
X  : (N, 3)
W1 : (m1, 3)
b1 : (1, m1) -> diffusé en (N, m1)
H  : (N, m1)
W2 : (m2, m1)
b2 : (1, m2) -> diffusé en (N, m2)
Y  : (N, m2)

## c (fait)
## d (fait)
## e (à vérifier!)
## f (à faire)
Tâche                   | Fonction finale (Sortie) | Fonction de perte (Loss)
------------------------|--------------------------|---------------------------
Classification binaire  | 1. ___________           | A. ___________
Classification multi    | 2. ___________           | B. ___________
Régression pure         | 3. Identité (aucune)     | C. MSE (Mean Squared Error)
# Ex 4: Votre premier réseau de neurones (∼45mn, – moyen)
## a
## b
## c
## d
## e
# Ex 5: Utilisation de TensorBoard (∼45mn, – moyen)
## a
## d
## e