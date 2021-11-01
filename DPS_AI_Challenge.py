import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV read
df = pd.read_csv("../../210619monatszahlenjuni2021monatszahlen2106verkehrsunfaelle.csv")#, usecols=["MONATSZAHL","AUSPRAEGUNG","JAHR","MONAT","WERT"])
#print(df['VORJAHRESWERT'])
#print(df['JAHR'])
#print(df)

data = df.values # convert to numpy array (2dim)

#print(data)
#print(df['JAHR'])
#print(data[:,3]) #month


# Put data in format
category = 'Alkoholunfälle'
type = 'insgesamt'

dataset = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] != 'Summe') & (df["WERT"].notnull()) & (df["VORJAHRESWERT"].notnull())]# & (df["JAHR"] != 2020)] # filter 

print(dataset)
print(dataset.shape)
print(data.shape)

sortdata = sorted(dataset, key=lambda x: x[3]) # sort by date
sortdata = np.array(sortdata)
sortdata = np.delete(sortdata, [0, 1], axis=1) # delete columns MONATSZAHL and AUSPRAEGUNG 

plt.scatter(sortdata[:,1], sortdata[:,2])
plt.show()