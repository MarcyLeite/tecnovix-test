import dataset
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

dataset.download()
df = dataset.import_as_df()

gender_map = { 'Male': 0, 'Female': 1 }

base_map = { 'No': 0, 'Yes': 1 }

phone_map = base_map.copy()
phone_map['No phone service'] = 0

internet_map = base_map.copy()
internet_map['No internet service'] = 0

def pre_process(df_or):
	df = df_or
	df.drop(columns=['customerID', 'Contract', 'PaperlessBilling', 'PaymentMethod'], inplace=True)

	df['gender'] = df['gender'].map(gender_map)
	df['Partner'] = df['Partner'].map(base_map)
	df['Dependents'] = df['Dependents'].map(base_map)
	df['PhoneService'] = df['PhoneService'].map(base_map)

	df['MultipleLines'] = df['MultipleLines'].map(phone_map)

	df = pd.get_dummies(df, columns=['InternetService'])

	df.rename(columns={'InternetService_Fiber optic': 'InternetService_FiberOptic'}, inplace=True)
	df['InternetService_DSL'] = df['InternetService_DSL'].astype(int)
	df['InternetService_FiberOptic'] = df['InternetService_FiberOptic'].astype(int)
	df.drop(columns=['InternetService_No'], inplace=True)

	df['InternetService'] = df.apply(lambda r: 1 if r['InternetService_DSL'] == 1 or r['InternetService_FiberOptic'] == 1 else 0, axis=1)

	df['OnlineSecurity'] = df['OnlineSecurity'].map(internet_map)
	df['OnlineBackup'] = df['OnlineBackup'].map(internet_map)
	df['DeviceProtection'] = df['DeviceProtection'].map(internet_map)
	df['TechSupport'] = df['TechSupport'].map(internet_map)
	df['StreamingTV'] = df['StreamingTV'].map(internet_map)
	df['StreamingMovies'] = df['StreamingMovies'].map(internet_map)
	
	df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
	df['Churn'] = df['Churn'].map(base_map)

	df.dropna(inplace=True)
	return df

df = pre_process(df)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()

def tune_model(model, X_train, y_train):
	param_grid = {
		'n_neighbors': range(1,50),
		'metric': ['euclidean', 'manhattan', 'minkowski'],
		'weights': ['uniform', 'distance']
	}
	
	grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
	grid_search.fit(X_train, y_train)
	return grid_search.best_estimator_

def evalutate_model(model, X_test, y_test):
	prediction = model.predict(X_test)
	accuracy = accuracy_score(y_test, prediction)
	matrix = confusion_matrix(y_test, prediction)
	return accuracy, matrix

models = {
  'Logistic Regression': LogisticRegression(max_iter=4000),
	'KNeighbors': KNeighborsClassifier(),
	'Random Forest': RandomForestClassifier(),
	'HistGB': HistGradientBoostingClassifier(),
	'SVC': SVC(),
}

def plot(name, accuracy, matrix):
	labels = ['Churn', 'Not Churn']
	fig, ax = plt.subplots()
	ax.imshow(matrix)

	ax.set_xticks(range(len(labels)), labels, ha="right")
	ax.set_xlabel('Prediction')
	ax.set_yticks(range(len(labels)), labels=labels)
	ax.set_ylabel('True Value')
	
	ax.set_title(f'Confusion Map - {name} ({accuracy})')

	for i in range(len(labels)):
		for j in range(len(labels)):
			ax.text(j, i, matrix[i, j],
				ha="center", va="center", color="w")

	fig.tight_layout()

for name, model in models.items():
	model.fit(X_train, y_train)
	accuracy, matrix = evalutate_model(model, X_test, y_test)

	percent = f'{accuracy * 100:.2f}%'

	print('-----------------------------------------')
	print(f'Accuracy ({name}): percent')
	plot(name, percent, matrix)

	plt.tight_layout()

	plt.savefig(f'images/matrix-{name.lower().replace(' ', '-')}.png')
	
