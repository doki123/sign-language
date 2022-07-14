from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import pandas as pd
import joblib 


df = pd.read_csv('world_landmark_hands/world_handtracking_data2.csv', index_col=[0])  # removes issue of 'Unnamed: 0' by treating it as index
print(df.shape)
# exit()
# df.drop(df.columns[0:2], axis=1, inplace=True)  # supposed to drop the two (why are there two???) weird empty columns at the beginning

# print(len(df.columns))
# print(df['CLASS'])

# df.drop('', axis=1, inplace=True)

# if df.columns[0] != 'WRIST_X':  # Note: This is for the 'Unnamed: 0' column that keeps appearing
#     df.drop(df.columns[0], axis=1, inplace=True)

# print(df)

x = df.drop('CLASS', axis=1)
y = df.CLASS

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x.values, y, random_state=4, train_size = 0.8, shuffle=True)


model = LogisticRegression(max_iter=1400, solver="sag")  # note: for higher datasate numbers, saga might be needed instead of sag
# for reference: https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

joblib.dump(model, 'world_landmark_hands/world_landmark_model.joblib')
# model.save()