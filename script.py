import pandas as pd
from ml import salvar_modelo, treinar_modelo

df = pd.read_csv('dataset.csv')

resultado = treinar_modelo(df)

print("Comparação de modelos (R² médio em validação cruzada):")
for nome, r2_cv in sorted(resultado.comparacao.items(), key=lambda item: item[1], reverse=True):
    destaque = " <- escolhido" if nome == resultado.nome_modelo else ""
    print(f"  {nome}: {r2_cv}%{destaque}")

print()
print(f"Modelo escolhido: {resultado.nome_modelo}")
print(f"R² médio (validação cruzada): {resultado.cv_r2_medio}%")
print(f"R² (conjunto de teste): {resultado.r2_teste}%")
print(f"MAE (conjunto de teste): R$ {resultado.mae_teste:,.2f}")
print(f"RMSE (conjunto de teste): R$ {resultado.rmse_teste:,.2f}")

salvar_modelo(resultado.pipeline)
print("\nModelo salvo em modelo_treinado.pkl")
