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

---------------------------------------------------
# Le projet - Music genre classification


* https://github.com/librosa/librosa (Pour trouver les features d’une nouvelle musique (MIR – Music Information Retrieval) i.e. récupérer l’empreinte d’une musique)
* https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
* [Lien vers son github](https://github.com/parulnith/Music-Genre-Classification-with-Python)



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

