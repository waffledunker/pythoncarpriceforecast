# TIME SERIES FORECASTING

import pandas as pd # used for importing data to program
from pandas import datetime  # need this to parse data 
import matplotlib.pyplot as plt

#FIRST PART START

def parser(x):
	return datetime.strptime(x,'%Y-%m')

sales = pd.read_csv('sales-cars.csv',index_col=0, parse_dates=[0], date_parser = parser) 
 #data imported from csv file,also parse data(we created a parser func above) with month as an index of data

##print(sales.head())  #prints head of imported data.

#print(sales.index[1]) #prints index=1,we used month as an index

#sales.plot() #first we plot the graph
###plt.show() #now we display it
#graph is NOT STATIONARY (stationarty means mean,variance and covariance is constant over periods)

#we need to convert series to stationary inorder to analyse and make forecast

sales_diff=sales.diff(periods=1)  #now we use diff method to calculate (t- (t-1))
#integrated order of 1 because we did differentiate 1 time,denoted by d for difference
#this is one of the parametere of ARIMA model.(I for integrated)

sales_diff = sales_diff[1:]  # we extracted NaN value out by starting index 1
#print(sales_diff.head()) #print sales_diff
#sales_diff.plot() #plot sales_diff

from statsmodels.graphics.tsaplots import plot_acf
# we need statsmodels to use plot_acf function
#plot_acf(sales) # auto-correlation func plot(ACF plots two signals which are t and t-1 and takes diff of them)
#plot_acf(sales_diff) # our data graph is now close to stationary as we see in this plot.values are rotating around x axis
#plt.show()  #displays two plots we created

X = sales.values
#print(X)  #our data for graph
#print(X.size)  #how many data we have

train = X[0:27] #train data is %90 of total = 27
test = X[26:]	#test data is %10 of total = 9
predictions = [] #predictions variable,those will be stored here
# now we have same variance,same mean through out graph it means it is stationary
#now we are done with analysis of data.

#AUTO REGRESSIVE AR MODEL
#FIRST PART ENDED



#SECOND PART START

from statsmodels.tsa.ar_model import AR  #auto regression
from sklearn.metrics import mean_squared_error  #mean_square_error 


model_ar = AR(train)  #initiate model and pass train data
model_ar_fit = model_ar.fit() #for better values
#we have finished training,now predictions

predictions = model_ar_fit.predict(start =26,end = 36) #predictions assigned to the variable
plt.plot(test) #plot test data
plt.plot(predictions,color = 'red') #plot pred data as red

# plt.show()  #display plots
#SECOND PART END

#THIRD PART START

# ARIMA MODEL

from statsmodels.tsa.arima_model import ARIMA
#p,d,q parameters must be specified 
#p=periods taken for autoregres model
#d = order of integration,difference
#q= periods in moving average model(lag)
model_arima = ARIMA(train,order =(9,1,0))
model_arima_fit = model_arima.fit()
#print(model_arima_fit.aic)

predictions = model_arima_fit.forecast(steps = 10)[0] 
#in arima instead of predict,we use forecast function
#print(predictions)

#plt.plot(test) #plot test data
plt.plot(predictions,color = 'yellow') #plot pred data as red

mean_square_error(test,predictions) #mean square error value should be minimum between selected parameters.

plt.show()

import warnings # to ignore upcoming warnings during execution of code
import itertools # import this to find best combination of arima order parameters

warnings.filterwarnings('ignore') #to ignore warnings
p=d=q= range(0,5)

pdq = list(itertools.product(p,d,q))
print(pdq)

'''  # for loop to try arima order param values
for param in pdq:  
	#what this for loop does is it tries to find best comb of parameters for our pred. graph
	try:
		model_arima = ARIMA(train,order=param)
		model_arima_fit = model_arima.fit()
		print(mean_square_error,param,model_arima_fit.aic)
	except:
		continue
'''


#THIRD PART END