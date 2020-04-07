# TODO

Je propose que les implementations des algos soient ecrits dans des scripts a part, qu'on importera dans le notebook. Ensuite, est-ce qu'on etudie toutes les methodes dans un seul notebook ? (a voir, pour l'instant j'ai nomme le notebook de la random forest "MT - Random Forest" MT pour model training, mais si on decide de tout faire sur le meme, on le renommera).
_______
## Groupe

1. Naive Bayes Classifier (Linear Classifier)
2. k-NN
3. SVM

Ce sont les algos qui me paraissent les plus intéressants pour notre problème ainsi que accesibles, le SVM est sûrement plus dur à coder
Voire https://en.wikipedia.org/wiki/Statistical_classification
_________
## Elise
* `MT - Random Forest`
    * Trouver les bons parametres (Comment fonctionne [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) ? et autre)
    * Est-ce que l'ajout de features a influence le resultat ?
    * Ajout de commentaires
* `randomforest.py`
    * dans predict, X en 2D array
* Possible de traiter des features qui sont des chaines de carac et non des int ?

Ce que j'ai fait pour le moment :

07/04 
* Transformé randomForest et decisionTree en `estimator` et revu le type des parametres des fonctions fit et predict
* 1ere etape Random Search CV avec une belle erreur ...
_________
## Telio
* Coder le Naive Bayes Classifier

# Le projet - Music genre classification

* https://github.com/librosa/librosa (Pour trouver les features d’une nouvelle musique (MIR – Music Information Retrieval) i.e. récupérer l’empreinte d’une musique)
* https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
* [Lien vers son github](https://github.com/parulnith/Music-Genre-Classification-with-Python)

## Optimisation des hyperparametres
Au lieu de definir au hasard les hyperparametres, on cherche a trouver la meilleure combinaison d'hyperparametres pour obtenir le meilleur modele, cad celui qui a la meilleure precision/prediction.
Tester une combinaison = entrainer le modele avec cette combinaison

En 2 etapes :
1. Random Search Cross Validation (`RandomizedSearchCV`) : tester un large choix de combinaisons qui ont été formées en tirant aléatoirement des valeurs dans une grille d'hyperparametres.
Permet de réduire l'intervalle de recherche pour chaque hyperparamètre.
2. Grid Search Cross Validation (`GridSearchCV`) : à ce stade, on a une idee plus precise sur quels intervalles chercher. On essaye cette fois-ci toutes les combinaisons possibles.
L'objectif étant ensuite de sélectionner celui qui a obtenu la meilleure performance.

Liens : 
* https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 (a lire absolument + notion de k-fold cv)
* https://stackoverflow.com/questions/19335165/what-is-the-difference-between-cross-validation-and-grid-search

Mais avant d'optimiser quoi que ce soit, il faut bien evidemment construire le modele et pas n'importe comment ! On doit créer un objet `estimator`. Pourquoi ? Pcq c'est le type de l'objet que prend `RandomizedSearchCV` et `GridSearchCV` en arguments (en plus d'autres tels que la grille d'hyperparametres, le k du k-fold cv etc.)

Super lien qui explique comment le creer http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/ ou celui de sklearn https://scikit-learn.org/stable/developers/develop.html.

En résumé, c'est une classe qui hérite de `BaseEstimator` et `ClassifierMixin` (tous deux dans `sklearn.base`) avec :
* sa fonction `__init__(self, et tous les hyperparametres avec une valeur par defaut)`. Attention ici à écrire `self.param1 = param1` et non `self.param = autreParam`
* `fit(self, X, y)` c'est la methode qui permet d'entrainer le modele (c'etait notre decisionTree et randomForest). Ici X {numpy array} les donnees d'entrainement non labellisees et y {numpy array} les labels. En pratique on appelera la methode de cette maniere `nomModele.fit(data_train, label_train)`
* `predict(self, X, y=None)` = classify. X ici c'est aussi un {numpy array} et c'est censé renvoyer l'ensemble des predictions pour cet array de donnees. Pour l'instant j'ai reduit X a un seul vecteur et je vais l'etendre a un 2D array.
* `score(self, params propre au modele pour calculer le score)`




## Proposition structure du projet
1.	Data Input
2.	Exploratory Data Analysis
3.	Feature Engineering (MIR)
4.	Dividing the data into training and test set
5.	Classification with several methods (Random Forest, Linear Regression, Naive Bayes, Logistic Regression, SVM?)
6.	Training time? and accuracy with test set (on pourra faire un ptit tableau comparatif)
7.	Test with own dataset?
8.	Music recommendation? (with own dataset?)

Exemple de projet super bien construit !
* https://github.com/miguelfzafra/Latest-News-Classifier/tree/master/0.%20Latest%20News%20Classifier


# Idées pour le projet

## Techniques qui me paraissent accessibles pour nous :
* Supervised learning -> Random Forest, Linear Regression, Naive Bayes, Logistic Regression, LASSO, Ridge Regression, SVM ?
* Clustering -> k-means
* Dimensionality reduction -> PCA

A mettre en perspective avec cette liste incroyable de dataset “classiques” avec spécification des “default tasks” :
* https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research

## Liens sur le thème de la musique
On peut faire de la classification, du clustering et même des recommandations en ajoutant des données sur les préférences d’utilisateurs.

### Million Song Dataset (MSDS)
Audio features from 1M different songs. _Text_
* http://millionsongdataset.com/
    * Enormement d'infos autour de l'artiste, la musique (ses caractéristiques audio)
    * Features deja extraits (energy, loudness, mode, beats confidence, tempo, time signature, ...)
    * Possibilite de telecharger une partie du jeu de donnees (10k chansons, 1.8 Go)
* http://www-personal.umich.edu/~yjli/content/projectreport.pdf
    * Système de recommandation. Trouver correlation entre un user et ses chansons prefs, et correlation entre les chansons

### Free Music Archive (FMA) Music Analysis Dataset
100k songs (343 days, 1TiB) with a hierarchy of 161 genres, metadata, user data, free-form text _(Text, MP3)_. **Classification, recommandation**
* https://github.com/mdeff/fma
* https://archive.ics.uci.edu/ml/datasets/FMA:+A+Dataset+For+Music+Analysis
* https://arxiv.org/pdf/1612.01840.pdf (research paper)
* https://www.groundai.com/project/fma-a-dataset-for-music-analysis/
* Pour télécharger : https://academictorrents.com/details/dba20c45d4d6fa6453a4e99d2f8a4817893cfb94 (torrent c’est plus simple)


### Geographical Original of Music Data Set (Musique du monde)
Audio features of music samples from different locations _(Text)_. **Geographical classification, clustering**
* https://archive.ics.uci.edu/ml/datasets/Geographical+Original+of+Music
    * Features deja extraits avec marsyas
* https://sci-hub.zone/10.1109/ICDM.2014.73 (research paper)
* https://github.com/marsyas/marsyas (Pour récupérer les features)

### GTZAN Genre Collection
1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.
* http://marsyas.info/downloads/datasets.html

## Des tutos
Des tutoriels qui me semble accessibles “mainly from scratch” et avec “pandas”:
* https://jeppbautista.wordpress.com/2019/01/27/theory-to-application-logistic-regression-from-scratch-using-python/ (logistic regression)
* https://jeppbautista.wordpress.com/2019/01/26/theory-to-application-linear-regression-from-scratch-using-python/ (linear regression)
* https://jeppbautista.wordpress.com/2019/02/02/theory-to-application-naive-bayes-classifier-for-sentiment-analysis-using-python/ (naive Bayes classifier for sentiment analysis)
