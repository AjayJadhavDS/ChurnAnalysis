import os
import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


transactionData = pd.read_csv("TransactionData.csv")
orderedData = pd.read_csv("OrderedData.csv")
transactionData = transactionData.loc[transactionData["DateOfBirth"] != "26.05.0184"]

transactionData.CreationDate = pd.to_datetime(transactionData.CreationDate)
transactionData.DateOfBirth = pd.to_datetime(transactionData.DateOfBirth)

transactionData['Age'] = datetime.datetime.today().year - transactionData['DateOfBirth'].dt.year
AgeDf = transactionData.drop_duplicates(['CustomerId'], keep='first')
AgeDf = AgeDf[['CustomerId', 'Gender', 'Age']]

keep = {'TransactionType': ["Debit", "PosDebit"], 'TransactionState': ["CONFIRMED"]}
transaction_typeTable = transactionData[transactionData[list(keep)].isin(keep).all(axis=1)]
transaction_typeTable = transaction_typeTable[['CustomerId', 'TransactionType', 'Amount']]
transaction_typeTable = transaction_typeTable.groupby(['CustomerId']).mean()
transaction_typeTable.Amount = transaction_typeTable.Amount.abs()
transaction_typeTable = transaction_typeTable.reset_index()
transaction_typeTable = transaction_typeTable.rename(columns={'Amount': 'meanAmount'})


sdAmount = transactionData[transactionData[list(keep)].isin(keep).all(axis=1)]
sdAmount = sdAmount[['CustomerId', 'TransactionType', 'Amount']]
sdAmount = sdAmount.groupby(['CustomerId']).std()
sdAmount.Amount = sdAmount.Amount.abs()
sdAmount = sdAmount.reset_index()
sdAmount = sdAmount.rename(columns={'Amount': 'sdAmount'})

TransStateTable = transactionData[['CustomerId', 'TransactionState']]
TransStateTable = pd.crosstab(index=TransStateTable['CustomerId'], columns=transactionData['TransactionState'],
                              margins=False)
TransStateTable['OrderNotDeclined'] = TransStateTable['CONFIRMED'] + TransStateTable['INITIAL']
TransStateTable['OrderDeclined'] = TransStateTable['CANCELED']
TransStateTable = TransStateTable.reset_index()
TransStateTable = TransStateTable[['CustomerId', 'OrderDeclined', 'OrderNotDeclined']]

OrderStateTable = orderedData[['CustomerId', 'OrderState']]
OrderStateTable = pd.crosstab(index=OrderStateTable['CustomerId'], columns=OrderStateTable['OrderState'], margins=True)
OrderStateTable['OrderNotDeclined'] = OrderStateTable['IN_PREPARATION'] + OrderStateTable['READY']
OrderStateTable['OrderDeclined'] = OrderStateTable['DECLINED']
OrderStateTable = OrderStateTable.reset_index()
OrderStateTable = OrderStateTable[['CustomerId', 'OrderDeclined', 'OrderNotDeclined']]

OrderState = TransStateTable.set_index('CustomerId').add(OrderStateTable.set_index('CustomerId'), fill_value=0).\
    reset_index()
OrderState = OrderState[OrderState.CustomerId.isin(AgeDf.CustomerId)]

productTypeTable = orderedData.loc[orderedData["OrderState"] == "READY"]
productTypeTable = pd.crosstab(index=productTypeTable['CustomerId'], columns=productTypeTable['ProductType'],
                               margins=True)
productTypeTable = productTypeTable.reset_index()
productTypeTable["OrderMultipleProduct"] = productTypeTable["OrderProductPackage"] + productTypeTable["OrderAddOn"]
productTypeTable = productTypeTable[productTypeTable.CustomerId.isin(AgeDf.CustomerId)]
productTypeTable = productTypeTable[['CustomerId', 'OrderSingleProduct', 'OrderMultipleProduct']]

pickupData = orderedData.loc[orderedData["OrderState"] == "READY"]
pickupData.PickUpTime = pd.to_datetime(pickupData.PickUpTime)
pickupData.loc[:, "Hour"] = pickupData.PickUpTime.dt.hour
pickupData.loc[:, "Minute"] = pickupData.PickUpTime.dt.minute
pickupData.loc[:, "HourMinute"] = pickupData['Hour'].astype(str) + '.' + pickupData['Minute'].astype(str)
pickupData.loc[:, "HourMinute"] = pickupData["HourMinute"].apply(pd.to_numeric)
pickupData.loc[:, "Session"] = pd.cut(pickupData.HourMinute, [0.0, 6.0, 12.0, 18.0, 23.59],
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening'])
pickupData = pd.crosstab(index=pickupData["CustomerId"], columns=pickupData["Session"], margins=True)
pickupData = pickupData.reset_index()

activeWeeks = transactionData[["CustomerId", "CreationYear", "CreationWeek", "TransactionState", "TransactionType"]]
keep = {'TransactionType': ["Debit", "PosDebit"], 'TransactionState': ["CONFIRMED"]}
activeWeeks = activeWeeks[transactionData[list(keep)].isin(keep).all(axis=1)]
activeWeeks['CreationWeek'][activeWeeks['CreationYear'] == 2016] = \
    activeWeeks['CreationWeek'][activeWeeks['CreationYear']==2016] + 53

activeWeeks = pd.crosstab(index=activeWeeks["CustomerId"], columns=activeWeeks["CreationWeek"], margins=True)
activeWeeks = activeWeeks.reset_index()
activeWeeks[53] = activeWeeks[53] + activeWeeks[106]
activeWeeks = activeWeeks.drop(106, axis=1)

temp = pd.DataFrame(np.where(activeWeeks.iloc[:, 1:(activeWeeks.shape[1] - 1)] >= 1, 1, 0))
activeWeeks = pd.concat([activeWeeks.iloc[:, 0], temp], axis=1)
activeWeeks["WeeksActive"] = activeWeeks.iloc[:, 1:(activeWeeks.shape[1] - 1)].sum(axis=1)
activeWeeks["ExistingCustomerChurn"] = activeWeeks.iloc[:, (activeWeeks.shape[1] - 6):(activeWeeks.shape[1] - 1)]\
    .sum(axis=1)
activeWeeks["ExistingCustomerChurn"][activeWeeks["ExistingCustomerChurn"] >= 1] = 1
activeWeeks = activeWeeks[["CustomerId", "WeeksActive", "ExistingCustomerChurn"]]
activeWeeks.head()

inactiveWeeks = pd.DataFrame()
inactiveWeeks["CustomerId"] = (transactionData.CustomerId[~transactionData.CustomerId.isin(activeWeeks.CustomerId)])\
    .unique()
inactiveWeeks["WeeksActive"] = 0
inactiveWeeks["ExistingCustomerChurn"] = 0

frames = [activeWeeks, inactiveWeeks]
totalWeeks = pd.concat(frames, axis=0)
totalWeeks = totalWeeks[totalWeeks.CustomerId.isin(AgeDf.CustomerId)]

frames = [totalWeeks, OrderState, sdAmount, transaction_typeTable, productTypeTable, AgeDf, pickupData]

finalData = reduce(lambda left, right: pd.merge(left, right, on='CustomerId', how="left"), frames)
finalData = finalData.drop('All', axis=1)

le = preprocessing.LabelEncoder()
le.fit(finalData.Gender)
finalData.Gender = le.transform(finalData.Gender)
print finalData.head()

Response = finalData.ExistingCustomerChurn
finalData = finalData.drop('ExistingCustomerChurn', axis=1)
finalData = finalData.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(finalData, Response, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
model = rf.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Confusion Matrix', confusion_matrix(y_test, y_predict))
print('Accuracy is', round(accuracy_score(y_test, y_predict)*100,2))

