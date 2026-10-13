# TP2 – Régularisation, optimisation et métriques

## Exercice 1

### Question 1

C'est une fuite de données. La moyenne et l'écart-type utilisés pour normaliser sont calculés sur tout le dataset, donc en partie sur les données de validation et de test. Des informations sur ces données se retrouvent ainsi dans le prétraitement de l'entraînement. Les performances mesurées sont alors trop optimistes, et le modèle risque de décevoir sur des données réellement nouvelles.
Il faut découper d'abord le dataset, faire le fit du scaler uniquement sur le train, puis appliquer ce même scaler au val et au test.

### Question 2

On utiliserait torch.utils.data.IterableDataset, qui lit les données en flux au lieu de tout charger en mémoire. On peut le combiner avec `num_workers` dans le `DataLoader` pour paralléliser la lecture.

## Exercice 2

Sortie du script `train.py` avec les deux réglages (10 epochs, SGD, lr = 0.01) :

![Résultats exercice 2](images/img.png)

### Question 1

| Réglage | Loss (epoch 1 → 10) | Précision (epoch 1 → 10) |
|---|---|---|
| L1 = 1e-4, L2 = 1e-3 | 0.8508 → 0.7427 | 0.6154 → 0.6691 |
| L1 = 0.1, L2 = 0 | 6.7158 → 1.6340 (bloquée dès l'epoch 2) | ≈ 0.50 (aucune progression) |

Avec `l1_lambda = 0.1`, la loss ne diminue plus après l'epoch 2 et reste figée à 1.6340. La précision oscille autour de 50 %, ce qui correspond au hasard sur un dataset à peu près équilibré. Le réseau n'apprend plus rien.

La pénalité L1 domine la loss totale : pour la réduire, l'optimiseur pousse tous les poids vers 0 plutôt que d'améliorer la prédiction. Le modèle devient trop contraint pour représenter les données. Ce phénomène s'appelle le **sous-apprentissage** (*underfitting*).

### Question 2

`weight_decay`.

### Question 3

- **L1** favorise des poids **parcimonieux**. Beaucoup de poids sont ramenés exactement à 0, ce qui revient à une sélection implicite de variables.
- **L2** **réduit l'amplitude de tous les poids** de façon uniforme. Ils deviennent petits mais rarement nuls, et la valeur est répartie sur l'ensemble des poids.
