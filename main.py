# This is a sample Python script.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import os
import json
#Setting Seaborn
from flask import Flask

app = Flask(__name__)

crms = pd.read_csv("/Users/jonathancecil/Downloads/01_District_wise_crimes_committed_IPC_2001_2014.csv")

# def hello_world():
#     return 'Hello World!'




sns.set()
def crimerate(crms,attributes):
    district = 'ADILABAD'
    percentagecrime =0
    val = 0
    dcrms = crms.loc[district]
    for attribute in attributes:
        x = round(districtscrime(district, attribute))
        plt.bar(dcrms["YEAR"] + 2, x, color='green')
        plt.bar(dcrms["YEAR"], dcrms[attribute], color = 'red', label = attribute)
        val = x/(max(dcrms[attribute]))
        plt.xlabel("YEAR")
        plt.ylabel(attribute)
        plt.title(attribute)
        plt.show()
        print("THE RELATIVE GRADE IN CRIME RATE IS : ", val)



def traveladvice(attributes):
    #print(attributes)
    district1 = 'BIZAPUR'
    district2 = 'ADILABAD'
    sumwomen1 = 0
    sumwomen2 = 0
    death1 = 0
    death2 = 0
    strls = []
    val1 =[]
    val2= []
    for attribute in attributes:
        val1.append(round(districtscrime(district1, attribute)))
        val2.append(round(districtscrime(district2, attribute)))
        strls.append(str(round(districtscrime(district1, attribute)))+" : "+str(round(districtscrime(district2, attribute))))
    print(district1, " : ", district2)
    for i in range(0,len(strls)):
        print(attributes[i])
        if (attributes[i] == 'KIDNAPPING & ABDUCTION' or attributes[i] == 'ATTEMPT TO MURDER' or attributes[
            i] == 'DOWRY DEATHS' or attributes[i]=='CAUSING DEATH BY NEGLIGENCE'):
            death1+= val1[i]
            death2+= val2[i]
        if(attributes[i]=='RAPE' or attributes[i]=='KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS' or attributes[i]=='ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY'):
            sumwomen1 += val1[i]
            sumwomen2 += val2[i]
        print(strls[i])
    if(sumwomen1>sumwomen2):
        print(district1," is not as Safe as ", district2, "FOR WOMEN AND CHILDREN in the Upcoming Years")
    elif(sumwomen2>sumwomen1):
        print(district2, " is not as Safe as ", district1, "FOR WOMEN AND CHILDREN in the Upcoming Years")
    if (death1 > death2):
        print(district1, " might have MORE DEATH RATE DUE TO CRIME THAN ", district2)
        print("BETTER MOVE TO ", district2)
    elif (death2> death1):
        print(district2, " might have MORE DEATH RATE DUE TO CRIME THAN", district1)
        print("BETTER MOVE TO", district1)

def pacfcomparision(df):
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(df["YEAR"], df["TOTAL IPC CRIMES"].diff());
    axes[0].set_title('1st Differencing')
    #axes[1].set(ylim=(0, 5))
    plot_pacf(df["TOTAL IPC CRIMES"].diff().dropna(),lags=len(df["TOTAL IPC CRIMES"])/3, ax=axes[1])

    plt.show()

def adfcomparision(df):
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=True)
    axes[0, 0].plot(df["YEAR"], df["TOTAL IPC CRIMES"]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df["TOTAL IPC CRIMES"], ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(df["YEAR"],df["TOTAL IPC CRIMES"].diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df["TOTAL IPC CRIMES"].diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(df["YEAR"],df["TOTAL IPC CRIMES"].diff().diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df["TOTAL IPC CRIMES"].diff().diff().dropna(), ax=axes[2, 1])

    plt.show()
    result = 1
    return result


def dickey_fuller(df):
    from statsmodels.tsa.stattools import adfuller
    from numpy import log
    result = adfuller(df["TOTAL IPC CRIMES"], autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1]>0.05:
        d = adfcomparision(df)
    else:
        d =0
    return d
def AUTOARIMA(x,train, test, a,fore=0 ):
    y = train[a]
    autoArima = pm.auto_arima(y,seasonal=True,trace=True, error_action='ignore', suppress_warnings=True)
    autoArima.fit(y)
    try:
        forecast= autoArima.predict(len(test['YEAR'])+fore)
    except ValueError:
        print('Skipped Auto Arima')
        return 10000
    if (fore > 0):
        z = x[x['YEAR'] >= 2010]
        forecast = pd.DataFrame(forecast, index=z['YEAR'] + fore, columns=['Predictions'])
    else:
        forecast = pd.DataFrame(forecast, index=test["YEAR"], columns=['Predictions'])

    plt.plot(forecast.index,forecast["Predictions"], color = 'Purple')
    if (fore == 0):
        autoArima_rmse = np.sqrt(mean_squared_error(test[a], forecast["Predictions"]))
        return autoArima_rmse
    return forecast["Predictions"]



def SARIMA(x,train, test,a,fore=0 ):
    y = train[a]
    SARIMAXmodel = SARIMAX(y, order=(3, 1, 4), seasonal_order=(2, 2, 2, 7),initialization='approximate_diffuse')
    SARIMAXmodel = SARIMAXmodel.fit()
    try:
        y_pred = SARIMAXmodel.get_forecast(len(test['YEAR'])+fore)
    except ValueError:
        return 10000
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])

    if(fore>0):
        z = x[x['YEAR']>=2010]
        y_pred_df.index = z['YEAR']+fore
    else:
        y_pred_df.index = test["YEAR"]
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_df.index,y_pred_out, color='Blue', label='SARIMA Predictions')
    if(fore ==0 ):
        sarima_rmse = np.sqrt(mean_squared_error(test[a], y_pred_df["Predictions"]))
        return sarima_rmse
    return y_pred_out


def ARIMA(x,train, test,a,fore=0):
    y = train[a]
    from statsmodels.tsa.arima.model import ARIMA
    # 1,1,2 ARIMA Model
    model = ARIMA(y, order=(0,0,1))
    model_fit = model.fit()
    #print(model_fit.summary())
    #predict = model_fit.predict(typ = 'levels')
    try:
      y_pred = model_fit.get_forecast(len(test["YEAR"])+fore)
    except ValueError:
        print('skipped')
        return 10000
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = model_fit.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    if (fore > 0):
        z = x[x['YEAR'] >= 2010]
        y_pred_df.index = z['YEAR'] + fore
    else:
        y_pred_df.index = test["YEAR"]
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_df.index, y_pred_out, color='Yellow', label='ARIMA Predictions')
    if (fore == 0):
        arima_rmse = np.sqrt(mean_squared_error(test[a], y_pred_df["Predictions"]))
        return arima_rmse
    return y_pred_out

def ARMA(x,train, test,a,fore=0):
    y = train[a]
    ARMAmodel = SARIMAX(y, order=(2, 0, 1),initialization='approximate_diffuse')
    ARMAmodel = ARMAmodel.fit()
    try:
        y_pred = ARMAmodel.get_forecast(len(test["YEAR"])+fore)
    except ValueError:
        print("Skipped")
        return 10000
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    if (fore > 0):
        z = x[x['YEAR'] >= 2010]
        y_pred_df.index = z['YEAR'] + fore
    else:
        y_pred_df.index = test["YEAR"]
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_df.index, y_pred_out, color='green', label='Predictions')
    if (fore == 0):
        arma_rmse = np.sqrt(mean_squared_error(test[a], y_pred_df["Predictions"]))
        return arma_rmse
    return y_pred_out

def modeling(dcrms,a):
    train = dcrms[dcrms['YEAR'] < 2012]
    test = dcrms[dcrms['YEAR'] >=2012]
    plot_setter(train,a, 'black')
    plot_setter(test,a, 'red')
    d =1#dickey_fuller(train)
    p =1
    q=2
    autoArima_rmse =AUTOARIMA(dcrms,train,test,a)
    arma_rmse =ARMA(dcrms,train,test, a)
    arima_rmse =ARIMA(dcrms,train,test,a)
    sarima_rmse = SARIMA(dcrms,train, test, a)
    print(autoArima_rmse, arma_rmse, arima_rmse, sarima_rmse)
    if autoArima_rmse<= min(arma_rmse, arima_rmse, sarima_rmse):
        plot_setter(train, a, 'black')
        plot_setter(test, a, 'red')
        print("AUTO ARIMA WAS SELECTED _____")
        result = AUTOARIMA(dcrms,train, test, a, 2)
        forecast = result.tail(1).item()
        print(forecast)
        plt.show()
    elif arma_rmse<=min(autoArima_rmse, arima_rmse, sarima_rmse):
        print("ARMA WAS SELECTED _____")
        plot_setter(train, a, 'black')
        plot_setter(test, a, 'red')
        result = ARMA(dcrms,train, test, a, 2)
        forecast = result.tail(1).item()
        print(forecast)
        plt.show()
    elif arima_rmse<=min(autoArima_rmse, arma_rmse, sarima_rmse):
        print("ARIMA WAS SELECTED _____")
        plot_setter(train, a, 'black')
        plot_setter(test, a, 'red')
        result = ARIMA(dcrms,train, test, a, 2)
        forecast = result.tail(1).item()
        print(forecast)
        plt.show()
    elif sarima_rmse<=min(autoArima_rmse, arima_rmse, arma_rmse):
        print("SARIMA WAS SELECTED _____")
        result = SARIMA(dcrms,train, test, a, 2)
        forecast =result.tail(1).item()
        print(forecast)
        plot_setter(train, a, 'black')
        plot_setter(test, a, 'red')
        plt.show()
    return forecast


def plot_setter(data,a, colour='black'):
    plt.ylabel(a)
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.plot(data['YEAR'], data[a], color = colour)


#district wise selection

def districtscrime(district, attribute, color= 'black'):
    dcrms = crms.loc[district]
    plot_setter(dcrms,attribute,color)
    result = modeling(dcrms,attribute)
    return result

def appwebpost():
    # selecting a district
    crms.set_index("DISTRICT", inplace=True)
    count = 0
    districts = []
    attributes = []
    for i in crms.index:
        districts.append(i)
    for i in crms:
        if (count > 31):
            break
        attributes.append(i)
        count += 1
    attributes.pop(0)
    attributes.pop(0)
    res = []
    for i in districts:
        if i not in res:
            res.append(i)
    print(res)



#main
#selecting a district
crms.set_index("DISTRICT", inplace= True)
count =0
districts = []
attributes = []
attribute = 'MURDER'
for i in crms.index:
    districts.append(i)
for i in crms:
    if(count>31):
        break
    attributes.append(i)
    count+=1
attributes.pop(0)
attributes.pop(0)
res = []
for i in districts:
    if i not in res:
        res.append(i)
districts = res
districtdict = dict()
# if os.path.exists('forecast.json'):
#     with open('forecast.json', 'r') as f:
#         districtdict = json.load(f)
#
# for i in range(len(districts)):
#     try:
#         district = districts[i]
#         if not districtdict.get(district ,None):
#             districtdict[district] = dict()
#             for attribute in attributes:
#                 if not districtdict[district].get(attribute, None):
#                     districtdict[district][attribute] = round(districtscrime(district, attribute))
#     except ValueError:
#         continue
#     if i % 5 == 0:
#         with open("forecast.json", "w") as f:
#             f.write(json.dumps(districtdict, indent=4))


#
traveladvice(attributes)
crimerate(crms, attributes)
