# SortingHatLib


1. Install the package using python-pip

```bash

git clone https://github.com/pvn25/SortingHatLib.git

pip install SortingHatLib/
```
2. Import the library using 

```bash

import sortinghat.pylib as pl

```
3. Choose from the available models:


```bash
# rf: Random Forest
# neural: Neural Model
pl.Initialize('rf')

```

4. Perform base featurization of the raw CSV file:

```bash

data1 = pl.BaseFeaturization('rawcsvfile.csv')

```

5. More featurization (n-gram feature extraction) for Classical ML models:

```bash

data2 = pl.Featurize(data1)

```

6. Finally, load the model for prediction

```bash

pl.LoadModel(data2)

```

The output is given as confidence scores:

| Column   | Inferred Feature Type | Confidence Score |
|----------|-----------------------|------------------|
| age      | Numeric               | 0.870804         |
| gender   | Categorical           | 0.768513         |
| bmi      | Numeric               | 0.640905         |
| children | Categorical           | 0.422998         |
| smoker   | Categorical           | 0.783468         |
| region   | Categorical           | 0.864832         |
| charges  | Not-Generalizable     | 0.485356         |


