# TP2 – Régularisation, optimisation et métriques

## Exercice 1

### Question 1

C'est une fuite de données. La moyenne et l'écart-type utilisés pour normaliser sont calculés sur tout le dataset, donc en partie sur les données de validation et de test. Des informations sur ces données se retrouvent ainsi dans le prétraitement de l'entraînement. Les performances mesurées sont alors trop optimistes, et le modèle risque de décevoir sur des données réellement nouvelles.
Il faut découper d'abord le dataset, faire le fit du scaler uniquement sur le train, puis appliquer ce même scaler au val et au test.

### Question 2

On utiliserait torch.utils.data.IterableDataset, qui lit les données en flux au lieu de tout charger en mémoire. On peut le combiner avec `num_workers` dans le `DataLoader` pour paralléliser la lecture.
