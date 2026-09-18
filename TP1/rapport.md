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
## a
## b
## c
## d
## e
## f
# Ex 3: Exercices théoriques (Papier & Markdown) (∼30mn, – moyen)
## a
## b
## c
## d
## e
## f
# Ex 4: Votre premier réseau de neurones (∼45mn, – moyen)
## a
## b
## c
## d
## e
# Ex 5: Utilisation de TensorBoard (∼45mn, – moyen)
## a
## b
## c
## d
## e