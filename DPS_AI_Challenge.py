import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from copy import copy, deepcopy


def CreateData(month, year, sortdata, currdata, model, poldeg):

    # Date format transformation 
    if month < 10:
        month = "0" + str(month)
    else:
        month = str(month)

    # Model input generation
    monat_val = "" + str(year) + month
    prev_year = "" + str(year-1) + month
    vjw_val = sortdata[sortdata[:,1] == prev_year, 3]

    tmp = [monat_val, vjw_val[0]]
    xnew_ = np.concatenate((tmp, currdata))
    xnew_[1] = float(xnew_[1])
    #print(xnew_)
    xnew_ = np.reshape(xnew_, (1, xnew_.shape[0]))
    #print(xnew_)

    # New data row creation
    xnew = PolynomialFeatures(degree=poldeg).fit_transform(xnew_)
    ynew = model.predict(xnew)

    appenddata = np.concatenate(([ ynew[0][0], vjw_val[0] ], currdata))
    appenddata = np.concatenate(([ year, monat_val ], appenddata))
    #print(appenddata)

    # Input data for next run calc
    vvp_val = (ynew[0][0] / sortdata[-1,2] - 1.0) * 100
    vvjm_val = (ynew[0][0] / sortdata[sortdata[:,1] == prev_year, 2] - 1.0) * 100
    ma_val = (ynew[0][0] + sum(sortdata[-11:,2])) / 12
    currdata = [vvp_val, vvjm_val, ma_val]
    #print(currdata)

    return (appenddata, currdata)


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
desyear = 2021
desmonth = 12

dataset = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] != 'Summe') & (df["WERT"].notnull()) & (df["VORJAHRESWERT"].notnull())]# & (df["JAHR"] != 2020)] # filter 

#print(dataset)
#print(dataset.shape)
#print(data.shape)

sortdata = sorted(dataset, key=lambda x: x[3]) # sort by date
sortdata = np.array(sortdata)
sortdata = np.delete(sortdata, [0, 1], axis=1) # delete columns MONATSZAHL and AUSPRAEGUNG 

#plt.scatter(sortdata[:,1], sortdata[:,2])
#plt.show()

tmp = sortdata[-1,4:7]
currdata = deepcopy(tmp)
#print(currdata)

sortdata[1:,4:7] = sortdata[0:-1,4:7] # shift data to next index, where it is known -> input parameter definition
#print(currdata)
sortdata = np.delete(sortdata, (0), axis=0) # delete first row
#plt.scatter(sortdata[:,3], sortdata[:,4])
#plt.show()

# Create model
x_ = sortdata[:,[1, 3, 4, 5, 6]]
y = sortdata[:,2]
y = y.reshape((y.shape[0],1))

#print(x.shape)
#print(y.shape)

# higher degree for forecasts in near future - overall more adaptive and no overfitting model / oversized values
if(desyear < 2022):
    poldeg = 3
else:
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
#print(result)

plt.figure(figsize=(16,8))
plt.plot(sortdata[:,1], result[:,0], 'g')
plt.plot(sortdata[:,1], result[:,1], 'r')
plt.show()


## Test model
#year = "2021"
#month = "9"
#if(len(month) == 1):
#    month = "0" + month
#
#newdata = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] == (year + month))]
##newdata = data[(df["MONATSZAHL"] == category) & (df["AUSPRAEGUNG"] == type) & (df["MONAT"] != 'Summe') & (df["JAHR"] == 2021)]
#newdata = np.array(newdata)
#xnew_ = newdata[:,[3, 5]]


#xnew = PolynomialFeatures(degree=poldeg).fit_transform(xnew_)
#ynew = model.predict(xnew)
#print(xnew)
#print(ynew)

#xrealplot = np.concatenate((sortdata[:,1],np.array(xnew_[:,0])))
#xpredplot = np.array(xnew_[:,0])
#yrealplot = np.concatenate((sortdata[:,2],newdata[:,4]))
#ypredplot = np.array(ynew[:,0])
#print(xnew_[:,0])
#print(yrealplot)
#print(ynew[:,0])

#plt.figure(figsize=(16,8))
#plt.plot(xrealplot, yrealplot, 'g')
#plt.plot(xpredplot, ypredplot, 'r')
#plt.show()


# Auto complete method

# Date format transformation 
desdate = "" + str(desyear)
if desmonth < 10:
    desdate += "0" + str(desmonth)

#print(desdate)

lastyear = int(sortdata[-1, 0])
lastmonth = int(sortdata[-1, 1]) - 100 * lastyear

# Generate forecasts row by row
if desyear < lastyear:
    returnvalue = yhat[sortdata[:,3] == desdate, 0]

elif desyear == lastyear:
    if desmonth <= lastmonth:
        returnvalue = yhat[sortdata[:,3] == desdate, 0]

    else:
        for j in range(lastmonth+1,desmonth+1):
            (appenddata, currdata) = CreateData(j,desyear,sortdata,currdata,model,poldeg)
            sortdata = np.vstack([sortdata, appenddata])

        returnvalue = sortdata[-1,2]

else:
    for j in range(lastmonth+1,12+1):
        (appenddata, currdata) = CreateData(j,lastyear,sortdata,currdata,model,poldeg)
        sortdata = np.vstack([sortdata, appenddata])

    for i in range(desyear-lastyear-1):
        for j in range(1,12+1):
            (appenddata, currdata) = CreateData(j,lastyear+i+1,sortdata,currdata,model,poldeg)
            sortdata = np.vstack([sortdata, appenddata])

    for j in range(1,desmonth+1):
        (appenddata, currdata) = CreateData(j,desyear,sortdata,currdata,model,poldeg)
        #print(sortdata)
        #print(appenddata)
        #print(sortdata.shape)
        sortdata = np.vstack([sortdata, appenddata])
        sortdata[-1,0] = float(sortdata[-1,0])
        #print(sortdata)
        #print(sortdata.shape)

    returnvalue = sortdata[-1,2]

print(returnvalue)

plt.figure(figsize=(16,8))
plt.plot(sortdata[:,1], sortdata[:,2], 'r')
plt.show()