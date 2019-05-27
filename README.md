# SortingHatLib


1. How to use the package?

```bash

git clone https://github.com/pvn25/SortingHatLib.git

pip install SortingHatLib/
```
2. Import the library using 

```bash

pl.Initialize('rf')

```
3. Choose from the available models:

rf: Random Forest
neural: Neural Model

Perform base featurization of the raw CSV file:

```bash

data1 = pl.BaseFeaturization('rawcsvfile.csv')

```

4. More featurization (n-gram feature extraction) for Classical ML models:

```bash

data2 = pl.Featurize(data1)

```

5. Finally, load the model for prediction

```bash

pl.LoadModel(data2)

```

