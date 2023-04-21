import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# dataset obtained from: https://www.kaggle.com/datasets/michaelbryantds/cpu-and-gpu-product-data

data = pd.read_csv("D:/cpu_gpu_data.csv")


# data.drop(columns="Unnamed: 0", inplace=True)
# data.drop(columns="Product", inplace=True)
# data.drop(columns="Release Date", inplace=True)
# data.drop(columns="Foundry", inplace=True)
# data.drop(columns="Vendor", inplace=True)

# x = data[['Freq (MHz)', 'Transistors (million)', 'Process Size (nm)']]
# data['Release Date'] = pd.to_datetime(data['Release Date'])
# data['Release Date'] = pd.to_numeric(data['Release Date'])
data['Transistors (million)'] = data['Transistors (million)'].fillna(0)
data['Process Size (nm)'] = data['Process Size (nm)'].fillna(0)

# predict the type (cpu or gpu based on the columns assigned to x)
# this is possibly the simplest case where there are only two choices
# I expect high levels of precision with what I am predicting here
x = data[['Freq (MHz)', 'Transistors (million)', 'Process Size (nm)']]
# x = data[['Freq (MHz)']]
y = data[['Type']]
# print(data.dtypes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.10, random_state=0)

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train.values.ravel())
predictions = lr.predict(x_test)
print()
print('Freq (MHz) Transistors (million) Process Size (nm) for predictions on Type')
print(classification_report(y_test, predictions))

# data is from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
data = pd.read_csv("D:/breast_cancer_data.csv")
# print(data.dtypes)

x = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
# predicting the diagnosis
y = data[['diagnosis']]
print(x.dtypes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.10, random_state=0)

lr = LogisticRegression(random_state=0, max_iter=500)
lr.fit(x_train, y_train.values.ravel())
predictions = lr.predict(x_test)

print()
print('stats on diagnosis predictive model (mean)')
print(classification_report(y_test, predictions))

x2 = data[['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]
# predicting the diagnosis
y2 = data[['diagnosis']]
# print(x.dtypes)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=.10, random_state=0)

lr2 = LogisticRegression(random_state=0, max_iter=500)
lr2.fit(x_train2, y_train2.values.ravel())
predictions2 = lr2.predict(x_test2)

print()
print('stats on diagnosis predictive model (worst)')
print(classification_report(y_test2, predictions2))