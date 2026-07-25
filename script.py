import csv
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from ml import salvar_modelo, treinar_modelo, verificar_saude_modelo

CAMINHO_HISTORICO = 'metrics_historico.csv'

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

r2_anterior = None
if os.path.exists(CAMINHO_HISTORICO):
    historico = pd.read_csv(CAMINHO_HISTORICO)
    if not historico.empty:
        r2_anterior = float(historico.iloc[-1]['r2_teste'])

problemas = verificar_saude_modelo(resultado.r2_teste, r2_anterior)

historico_existe = os.path.exists(CAMINHO_HISTORICO)
with open(CAMINHO_HISTORICO, 'a', newline='') as f:
    escritor = csv.writer(f)
    if not historico_existe:
        escritor.writerow(['timestamp', 'modelo', 'cv_r2_medio', 'r2_teste', 'mae_teste', 'rmse_teste'])
    escritor.writerow([
        datetime.now(timezone.utc).isoformat(),
        resultado.nome_modelo,
        resultado.cv_r2_medio,
        resultado.r2_teste,
        resultado.mae_teste,
        resultado.rmse_teste,
    ])

if problemas:
    print("\nALERTA DE QUALIDADE DO MODELO (possível drift nos dados):")
    for problema in problemas:
        print(f"  - {problema}")
    print("\nModelo NÃO foi salvo; mantendo o modelo_treinado.pkl anterior.")
    sys.exit(1)

salvar_modelo(resultado.pipeline)
print("\nModelo salvo em modelo_treinado.pkl")
