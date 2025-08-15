import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('dataset.csv')

modelo = linear_model.LinearRegression()

x = df[['m2']]
y = df[['valor']]

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y)

modelo.fit(x_treino,y_treino)

previsoes = modelo.predict(x_teste)

acuracidade =  round(r2_score(y_teste,previsoes) * 100,2)
print(f"A acuracidade do modelo é de {acuracidade}%")




with open("modelo_treinado.pkl","wb") as file:
    pickle.dump(modelo,file)