import os
import pandas as pd

# CSV read
df = pd.read_csv("../../210619monatszahlenjuni2021monatszahlen2106verkehrsunfaelle.csv")#, usecols=["MONATSZAHL","AUSPRAEGUNG","JAHR","MONAT","WERT"])
#print(df['VORJAHRESWERT'])
#print(df['JAHR'])
#print(df)

data = df.values # convert to numpy array (2dim)

#print(data)
#print(df['JAHR'])
#print(data[:,3]) #month

category = 'Alkoholunfälle'
type = 'insgesamt'

dataset = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] != 'Summe') & (df["WERT"].notnull()) & (df["VORJAHRESWERT"].notnull())]# & (df["JAHR"] != 2020)] # filter 

print(dataset)
print(dataset.shape)
print(data.shape)