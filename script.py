import pandas as pd
from ml import treinar_modelo, salvar_modelo

df = pd.read_csv('dataset.csv')

modelo, acuracidade = treinar_modelo(df)

print(f"Acurácia do modelo (R²): {acuracidade}%")
print(f"Coeficientes: m²={modelo.coef_[0][0]:.2f}, bairro={modelo.coef_[0][1]:.2f}")
print(f"Intercepto: {modelo.intercept_[0]:.2f}")

salvar_modelo(modelo)
print("Modelo salvo em modelo_treinado.pkl")
