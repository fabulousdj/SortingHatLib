# SortingHatLib

Library for ML feature type inference: https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/MLFeatureTypeInference

Original repo: https://github.com/pvn25/SortingHatLib.git

1. Install the package using python-pip

```bash

git clone https://github.com/fabulousdj/SortingHatLib.git

pip install SortingHatLib/
```
2. Import the library using 

```bash

import sortinghat.pylib as pl

```
3. Read in csv file using pandas


```bash
# rf: Random Forest, neural: Neural Model, knn: K-nn, logreg: Logistic Regression, svm: RBF_SVM
dataDownstream = pd.read_csv('adult.csv')

```

4. Perform base featurization of the raw CSV file:

```bash

dataFeaturized = pl.FeaturizeFile(dataDownstream)

```

5. bigram feature extraction for Random Forest:

```bash

dataFeaturized1 = pl.FeatureExtraction(dataFeaturized)

```

6. Finally, load the model for prediction

```bash

y_RF = pl.Load_RF(dataFeaturized1)

```



