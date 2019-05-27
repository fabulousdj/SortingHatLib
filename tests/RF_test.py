import sortinghat.pylib as pl

pl.Initialize('rf')
data1 = pl.BaseFeaturization('insurance.csv')
data2 = pl.Featurize(data1)
pl.LoadModel(data2)