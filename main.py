import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
print(data.describe())


def give_names_to_indices(ind):
    if ind == 0:
        return 'Extremely Weak'
    elif ind == 1:
        return 'Weak'
    elif ind == 2:
        return 'Normal'
    elif ind == 3:
        return 'OverWeight'
    elif ind == 4:
        return 'Obesity'
    elif ind == 5:
        return 'Extremely Obese'


data['Index'] = data['Index'].apply(give_names_to_indices)

sns.lmplot(x='Height', y='Weight', data=data, hue='Index', height=7, aspect=1, fit_reg=False)

people = data['Gender'].value_counts()

categories = data['Index'].value_counts()

# STATS FOR MEN
data[data['Gender'] == 'Male']['Index'].value_counts()

# STATS FOR WOMEN
data[data['Gender'] == 'Female']['Index'].value_counts()

data2 = pd.get_dummies(data['Gender'])
data.drop('Gender', axis=1, inplace=True)
data = pd.concat([data, data2], axis=1)

y = data['Index']
data = data.drop(['Index'], axis=1)

scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=101)

param_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, verbose=3)

grid_cv.fit(X_train, y_train)

print(grid_cv.best_params_)
# weight category prediction
pred = grid_cv.predict(X_test)

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print('Acuuracy is --> ', accuracy_score(y_test, pred) * 100)
print('\n')


def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]

    if gender == 'Male':
        details = np.array([[float(height), float(weight), 0.0, 1.0]])
    elif gender == 'Female':
        details = np.array([[float(height), float(weight), 1.0, 0.0]])

    y_pred = grid_cv.predict(scaler.transform(details))
    return (y_pred[0])


def recommend_diet(weight_category, gender):
    gender = gender.lower()

    if weight_category == 'Extremely Weak':
        if gender == 'male':
            return (
                "You should increase your daily calorie intake by consuming calorie-dense and protein-rich foods such as lean meats, dairy products, peanut butter, and healthy oils. "
                "Aim for 5-6 meals per day, and include strength training to help gain muscle mass.")
        elif gender == 'female':
            return (
                "Focus on a calorie-rich diet with foods like nuts, dairy, whole grains, and healthy fats. Try to eat 5-6 times a day and include moderate resistance training to build healthy weight.")

    elif weight_category == 'Weak':
        if gender == 'male':
            return (
                "Increase your portions slightly and add more carbohydrates like rice, pasta, and whole grains. Include snacks like granola bars and yogurt. Don't forget to hydrate well.")
        elif gender == 'female':
            return (
                "Aim for nutrient-dense meals including whole grains, beans, legumes, and eggs. Snack on nuts and fruits. Try to eat 4-5 meals a day and include light physical activity.")

    elif weight_category == 'Normal':
        return (
            "Maintain a balanced diet that includes a variety of fruits, vegetables, lean proteins (like fish or chicken), whole grains, and healthy fats. "
            "Drink enough water and engage in regular physical activity like walking or jogging.")

    elif weight_category == 'OverWeight':
        if gender == 'male':
            return (
                "Focus on portion control and choose high-fiber foods like vegetables, legumes, and oats. Avoid sugary drinks and processed snacks. Try to be physically active for at least 30 minutes a day.")
        elif gender == 'female':
            return (
                "Include more vegetables, fruits, and lean proteins in your meals. Limit processed carbohydrates and sugary items. Consistent, moderate-intensity exercise like swimming or brisk walking is beneficial.")

    elif weight_category == 'Obesity':
        if gender == 'male':
            return (
                "Reduce your calorie intake with a focus on high-fiber, low-fat foods. Avoid fast food and sugary beverages. Consider tracking your meals and consulting a dietitian for support.")
        elif gender == 'female':
            return (
                "Shift to a structured, portion-controlled diet. Focus on vegetables, legumes, whole grains, and low-fat dairy. Start with light exercise and gradually increase intensity.")

    elif weight_category == 'Extremely Obese':
        return (
            "It's strongly recommended to consult a healthcare provider. A structured, medically supervised diet plan may be necessary. "
            "Focus on natural, unprocessed foods, portion control, and begin with low-impact exercises like walking or water aerobics.")

    else:
        return "Invalid weight category or gender. Please check your input."
g = input("gender:")
h = int(input("height:"))
w = int(input("weighgt:"))
your_details = [g, h, w]
print("weight category:",lp(your_details))
print("recommend_diet:",recommend_diet(lp(your_details),g))
