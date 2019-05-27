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

rf: Random Forest
neural: Neural Model

```bash
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

