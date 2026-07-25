import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES_NUMERICAS = ['m2', 'quartos', 'vagas']
FEATURE_CATEGORICA = 'bairro'
ALVO = 'valor'

BAIRROS = ['Zona 1', 'Zona 2', 'Zona 3', 'Zona 4', 'Zona 5', 'Zona 6', 'Zona 7', 'Zona 8']

MODELOS_CANDIDATOS = {
    'Regressão Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
}


@dataclass
class ResultadoTreino:
    """Resultado da seleção de modelo: melhor pipeline treinado e suas métricas."""
    pipeline: Pipeline
    nome_modelo: str
    cv_r2_medio: float
    r2_teste: float
    mae_teste: float
    rmse_teste: float
    comparacao: dict = field(default_factory=dict)


def _construir_pipeline(modelo) -> Pipeline:
    pre_processador = ColumnTransformer([
        ('bairro', OneHotEncoder(handle_unknown='ignore'), [FEATURE_CATEGORICA]),
    ], remainder='passthrough')

    return Pipeline([
        ('pre_processamento', pre_processador),
        ('modelo', modelo),
    ])


def treinar_modelo(df: pd.DataFrame, cv_folds: int = 5) -> ResultadoTreino:
    """Compara modelos via validação cruzada, escolhe o de maior R² médio e o avalia
    em um conjunto de teste isolado (R², MAE e RMSE)."""
    colunas = FEATURES_NUMERICAS + [FEATURE_CATEGORICA]
    x = df[colunas]
    y = df[ALVO]

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

    comparacao = {}
    melhor_nome, melhor_pipeline, melhor_cv = None, None, -np.inf

    for nome, modelo in MODELOS_CANDIDATOS.items():
        pipeline = _construir_pipeline(modelo)
        scores = cross_val_score(pipeline, x_treino, y_treino, cv=cv_folds, scoring='r2')
        media = scores.mean()
        comparacao[nome] = round(media * 100, 2)

        if media > melhor_cv:
            melhor_nome, melhor_pipeline, melhor_cv = nome, pipeline, media

    melhor_pipeline.fit(x_treino, y_treino)
    previsoes = melhor_pipeline.predict(x_teste)
    rmse = mean_squared_error(y_teste, previsoes) ** 0.5

    return ResultadoTreino(
        pipeline=melhor_pipeline,
        nome_modelo=melhor_nome,
        cv_r2_medio=round(melhor_cv * 100, 2),
        r2_teste=round(r2_score(y_teste, previsoes) * 100, 2),
        mae_teste=round(mean_absolute_error(y_teste, previsoes), 2),
        rmse_teste=round(rmse, 2),
        comparacao=comparacao,
    )


def salvar_modelo(pipeline: Pipeline, caminho: str = 'modelo_treinado.pkl') -> None:
    """Serializa o pipeline treinado (pré-processamento + modelo) em disco."""
    with open(caminho, 'wb') as f:
        pickle.dump(pipeline, f)


def carregar_modelo(caminho: str = 'modelo_treinado.pkl') -> Pipeline:
    """Carrega e retorna o pipeline serializado."""
    with open(caminho, 'rb') as f:
        return pickle.load(f)


def preparar_dados(metragem: float, quartos: int, vagas: int, bairro: str) -> pd.DataFrame:
    """Converte inputs do usuário em DataFrame pronto para predição."""
    return pd.DataFrame({
        'm2': [metragem],
        'quartos': [quartos],
        'vagas': [vagas],
        'bairro': [bairro],
    })


def prever_valor(pipeline: Pipeline, metragem: float, quartos: int, vagas: int, bairro: str) -> float:
    """Retorna o valor previsto do imóvel em reais."""
    dados = preparar_dados(metragem, quartos, vagas, bairro)
    return float(pipeline.predict(dados)[0])
