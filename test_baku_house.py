import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import re
import time

house_dataset = pd.read_excel('processed_file.xlsx')

house_dataset['repair'] = house_dataset['repair'].replace(r'^\s*$', 'yoxdur', regex=True)
house_dataset['repair'] = house_dataset['repair'].fillna('yoxdur')

house_dataset.replace({'repair':{'var':0, 'yoxdur':1}}, inplace=True)
house_dataset.replace({'title_deed':{'var':0, 'yoxdur':1}}, inplace=True)
house_dataset.replace({'category':{'yeni':0, 'kohne':1}}, inplace=True)

house_dataset['region'] = house_dataset['region'].str.replace(r'^Nerimanov$', 'Neriman Nerimanov', regex=True)
house_dataset['price_1m2'] = house_dataset['price_1m2'].str.replace('AZN/m²', '', regex=False).str.replace(' ', '').astype(int)
house_dataset['area'] = house_dataset['area'].str.replace('m²', '', regex=False).str.replace(' ', '').str.split('.').str[0].astype(int)
house_dataset['price'] = house_dataset['price'].str.replace(' ', '').astype(int)
house_dataset['room_number'] = house_dataset['room_number'].astype(int)

X = house_dataset.drop(['currency', 'title', 'address', 'region', 'price', 'price_1m2'], axis=1)
Y = house_dataset['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

training_data_predict_lr = lin_reg_model.predict(X_train)
test_data_predict_lr = lin_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_train, training_data_predict_lr)
print("R squared Error : ", error_score)

test_error_score = metrics.r2_score(Y_test, test_data_predict_lr)
print("Test R squared Error : ", test_error_score)

#print(house_dataset.head(50))

def predict_house_price():
    area = int(input("Enter the area of your house(m2): "))
    room_number = int(input("Enter the number of rooms: "))
    title_deed = int(input("Does your house has title deed(yes-0/no-1): "))
    building_struct = int(input("New or old building structure(new-0/old-1): "))
    repair = int(input("Did you repaired your house ever(yes-0/no-1): "))

    input_data = pd.DataFrame([[building_struct, area, title_deed, repair, room_number]], columns=X.columns)
    price = lin_reg_model.predict(input_data)[0]
    print("Estimating your house marketing price...")
    time.sleep(5)
    print(f"\nPredicted Selling Price: {price:.0f} AZN")
    print(f"\n{price/area:.0f} AZN/m²")

predict_house_price()