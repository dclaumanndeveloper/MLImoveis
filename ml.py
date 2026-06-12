import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BAIRRO_ENCODING = {'Zona 3': 0, 'Zona 7': 1}


def treinar_modelo(df: pd.DataFrame) -> tuple:
    """Treina regressão linear com m² e bairro; retorna (modelo, acurácia_%)."""
    df = df.copy()
    df['bairro_cod'] = df['bairro'].map(BAIRRO_ENCODING)

    x = df[['m2', 'bairro_cod']]
    y = df[['valor']]

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=42)

    modelo = linear_model.LinearRegression()
    modelo.fit(x_treino, y_treino)

    previsoes = modelo.predict(x_teste)
    acuracidade = round(r2_score(y_teste, previsoes) * 100, 2)

    return modelo, acuracidade


def salvar_modelo(modelo, caminho: str = 'modelo_treinado.pkl') -> None:
    """Serializa o modelo treinado em disco."""
    with open(caminho, 'wb') as f:
        pickle.dump(modelo, f)


def carregar_modelo(caminho: str = 'modelo_treinado.pkl'):
    """Carrega e retorna o modelo serializado."""
    with open(caminho, 'rb') as f:
        return pickle.load(f)


def preparar_dados(metragem: float, bairro: str) -> pd.DataFrame:
    """Converte inputs do usuário em DataFrame pronto para predição."""
    bairro_cod = BAIRRO_ENCODING.get(bairro, 0)
    return pd.DataFrame({'m2': [metragem], 'bairro_cod': [bairro_cod]})


def prever_valor(modelo, metragem: float, bairro: str) -> float:
    """Retorna o valor previsto do imóvel em reais."""
    dados = preparar_dados(metragem, bairro)
    return modelo.predict(dados)[0][0]
