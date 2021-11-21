# SortingHatLib

Library for ML feature type inference: https://github.com/pvn25/MLDataPrepZoo/tree/master/MLFeatureTypeInference

Due to git-lfs limits, the resources files are moved to: https://drive.google.com/drive/folders/1eC8F5pO2hSoQf4RQM7zww49y2ZbLIvqG

By default, these resources will be auto downloaded the first time you run the program. If for some reason, this does not work you can try manual download.

1. Install the package using python-pip

```bash

git clone https://github.com/pvn25/SortingHatLib.git

pip install SortingHatLib/
```
2. Import the library using 

```bash

import sortinghat.pylib as pl

```
3. Read in csv file using pandas


```bash
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



