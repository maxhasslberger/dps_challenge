import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

#print(dataset)
#print(dataset.shape)
#print(data.shape)

sortdata = sorted(dataset, key=lambda x: x[3]) # sort by date
sortdata = np.array(sortdata)
sortdata = np.delete(sortdata, [0, 1], axis=1) # delete columns MONATSZAHL and AUSPRAEGUNG 

#plt.scatter(sortdata[:,1], sortdata[:,2])
#plt.show()



# Create model
x_ = sortdata[:,[1, 3]]
y = sortdata[:,2]
y = y.reshape((y.shape[0],1))

#print(x.shape)
#print(y.shape)

poldeg = 2
x = PolynomialFeatures(degree=poldeg).fit_transform(x_)

model = LinearRegression().fit(x, y)
score = model.score(x, y)

print("coefficient of determination:", score)

yhat = model.predict(x)
yhat = np.array(yhat)
result = np.zeros([sortdata.shape[0],2])
result[:,0] = sortdata[:,2]
result[:,1] = yhat[:,0]
print(result)

plt.figure(figsize=(16,8))
plt.plot(sortdata[:,1], result[:,0], 'g')
plt.plot(sortdata[:,1], result[:,1], 'r')
plt.show()


# Test model
year = "2021"
month = "9"
if(len(month) == 1):
    month = "0" + month

newdata = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] == (year + month))]
#newdata = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] != 'Summe') & (df["JAHR"] == 2021)]
newdata = np.array(newdata)
xnew_ = newdata[:,[3, 5]]


xnew = PolynomialFeatures(degree=poldeg).fit_transform(xnew_)
ynew = model.predict(xnew)
print(xnew)
print(ynew)

xrealplot = np.concatenate((sortdata[:,1],np.array(xnew_[:,0])))
xpredplot = np.array(xnew_[:,0])
yrealplot = np.concatenate((sortdata[:,2],newdata[:,4]))
ypredplot = np.array(ynew[:,0])
print(xnew_[:,0])
print(yrealplot)
print(ynew[:,0])

plt.figure(figsize=(16,8))
plt.plot(xrealplot, yrealplot, 'g')
plt.plot(xpredplot, ypredplot, 'r')
plt.show()