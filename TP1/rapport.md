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

![CUDA_false.png](images/CUDA_false.png)

On a probablement ce message d'erreur car ___build CPU-only___ a été installé ou que le GPU est non alloué par Slurm.

## f

![tensorboard.png](images/tensorboard.png)
___pip show tensorboard___

# Ex 3: Exercices théoriques (Papier & Markdown) (∼30mn, – moyen) (à faire en dehors de TSP)
## a
![3.a.jpg](images/3.a.jpg)

Sans le biais:

$$ Y = X W_1^T $$

$$ Z = Y W_2^T $$

Soient $X \in \mathbb{R}^{1 \times 3}$, $Y \in \mathbb{R}^{1 \times 4}$ et $Z \in \mathbb{R}^{1 \times 2}$.

On a donc $W_1 \in \mathbb{R}^{4 \times 3}$ et $W_2 \in \mathbb{R}^{2 \times 4}$

<br><br>


*Couche 1 et cachée*: $4 \times 3$ = 12 poids

*Couche cachée et de sortie*: $4 \times 2$ = 8 poids


<br><br>

Il y a **20** paramètres sans biais.

<br><br>

Pour la couche cachée, le biais est une matrice qui appartient à $\mathbb{R}^{1 \times 4}$

Pour la couche cachée, le biais est une matrice qui appartient à $\mathbb{R}^{1 \times 2}$

<br><br>

Soient **26** paramètres avec biais.

## b

$$H = ReLU( X W_1^T + b_1 )$$

$$Y = H W_2^T + b_2$$

<br><br>

Soient m1, m2 dans R

Dimensions :

X  : (N, 3)

W1 : (m1, 3)

b1 : (1, m1) -> diffusé en (N, m1)

H  : (N, m1)

W2 : (m2, m1)

b2 : (1, m2) -> diffusé en (N, m2)

Y  : (N, m2)

## c

![3.c.jpg](images/3.c.jpg)

$q = \frac{x}{y}$

$f = q + z$

<br><br>

*Forward pass*

$q = \frac{2}{4}$


$q = 0.5$

<br><br>
$f = 0.5 + 0$

**$f = 0.5$**

<br><br>

*Backpropagation*

$\frac{\partial f}{\partial f} = 1$

$\frac{\partial f}{\partial q} = 1$

$\frac{\partial f}{\partial z} = 1$

<br><br>

$\frac{\partial q}{\partial x} = \frac{1}{y}$

$\frac{\partial q}{\partial y} = - \frac{x}{y^2}$

<br>

$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x} = \frac{1}{y}$

$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial y} = - \frac{x}{y^2}$

## d

Soient: 

$x' = x - \eta \frac{\partial f}{\partial x}$

$y' = y - \eta \frac{\partial f}{\partial y}$

$z' = z - \eta \frac{\partial f}{\partial z}$

<br><br>

$x' = 2 - \frac{1}{4} = \frac{7}{4}$

$y' = 4 + \frac{1}{8} = \frac{33}{8}$

$z' = -1$

<br><br>

$f' = \frac{x'}{y'} + z'$

$f' = \frac{7 \times 8}{4 \times 33} - 1$

$f' = - \frac{19}{33}$

$f = \frac{1}{2}$  donc $f' < f$

La valeur de la fonction a diminué comme attendu.

## e (à vérifier!)
## f (à faire)
Tâche                   | Fonction finale (Sortie) | Fonction de perte (Loss)
------------------------|--------------------------|---------------------------
Classification binaire  | 1. ___________           | A. ___________
Classification multi    | 2. ___________           | B. ___________
Régression pure         | 3. Identité (aucune)     | C. MSE (Mean Squared Error)
# Ex 4: Votre premier réseau de neurones (∼45mn, – moyen)
## a (à faire) Expliquez brièvement à quoi servent les arguments batch_size et shuffle dans le DataLoader. Pourquoi shuffle doit-il avoir une valeur différente pour l'entraînement et pour le test ?
## b (à faire)
Dans la méthode forward, pourquoi utilise-t-on torch.flatten(x, 1) avant de passer les données à la couche linéaire ?

Pourquoi est-il crucial de ne pas ajouter de fonction d'activation Softmax à la fin de notre réseau quand on s'apprête à utiliser nn.CrossEntropyLoss dans PyTorch ?

## c (à faire)

Quelle est la différence fondamentale entre optimizer.zero_grad() et loss.backward() ?

## d

Pourquoi utilise-t-on le bloc with torch.no_grad(): lors de l'évaluation ? Quel est l'avantage en termes de ressources matérielles ?

Si votre classificateur prédisait les classes de manière purement aléatoire, à quelle précision (accuracy) environ devriez-vous vous attendre sur le jeu de test CIFAR-10 ?


# Ex 5: Utilisation de TensorBoard (∼45mn, – moyen)
## a

Pourquoi est-il important d'inclure la date, l'heure et les hyperparamètres dans le nom du dossier de logs (run_name) ?

## d
## e