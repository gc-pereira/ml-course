import pandas as pd

df = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv")
columns = df.columns

x = df[columns[0:3]]
y = df[columns[3]]

print(df.head())
print(x.head())
print(y.head())

x_train = x[:75]
x_test = x[75:]
y_train = y[:75]
y_test = y[75:]

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = 20,
                                                    stratify = y)

model = LinearSVC()
model.fit(x_train, y_train)

forecast = model.predict(x_test)
print('-------------------------------------------------')
accuracy = accuracy_score(y_test, forecast)
print(f'the algorithm has accuracy of {accuracy}')
print('-------------------------------------------------')
print(f'The size of train predict variable sample is {x_train.shape[0]}\nand an size of test is {x_test.shape[0]}')
print('-------------------------------------------------')
print(y_test.value_counts())
print('-------------------------------------------------')