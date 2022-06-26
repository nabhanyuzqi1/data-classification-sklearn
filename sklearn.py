
# machine learning classification
import sklearn


# For data manipulation
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn

#Step 2: Fetch data

import yfinance as yf
Df = yf.download('SPY', start="2012-01-01", end="2017-10-01")        
Df = Df.dropna()
Df.Close.plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()


#Step 3: Determine the target variable
y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)

#Step 4: Creation of predictors variables

Df['Open-Close'] = Df.Open - Df.Close
Df['High-Low'] = Df.High - Df.Low
X=Df[['Open-Close','High-Low']]

#Step 5: Test and train dataset split

split_percentage = 0.8
split = int(split_percentage*len(Df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]


#Step 6: Create the machine learning classification model using the train dataset

cls = SVC().fit(X_train, y_train)

#Step 7: The classification model accuracy

accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

#Step 8: Prediction

Df['Predicted_Signal'] = cls.predict(X)

# Calculate log returns
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()
