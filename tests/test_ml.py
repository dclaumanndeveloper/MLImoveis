import os

import pandas as pd
import pytest

from ml import (
    BAIRROS,
    ResultadoTreino,
    carregar_modelo,
    prever_valor,
    preparar_dados,
    salvar_modelo,
    treinar_modelo,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    caminho = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
    return pd.read_csv(caminho)


@pytest.fixture(scope="module")
def resultado(dataset: pd.DataFrame) -> ResultadoTreino:
    return treinar_modelo(dataset)


def test_treinar_modelo_retorna_resultado_valido(resultado: ResultadoTreino):
    assert resultado.pipeline is not None
    assert resultado.nome_modelo in {"Regressão Linear", "Random Forest", "Gradient Boosting"}
    assert resultado.r2_teste > 80
    assert resultado.mae_teste > 0
    assert resultado.rmse_teste > 0
    assert set(resultado.comparacao) == {"Regressão Linear", "Random Forest", "Gradient Boosting"}


def test_preparar_dados_gera_colunas_esperadas():
    df = preparar_dados(120, 3, 2, "Zona 1")
    assert list(df.columns) == ["m2", "quartos", "vagas", "bairro"]
    assert df.shape == (1, 4)
    assert df.iloc[0]["bairro"] == "Zona 1"


def test_prever_valor_retorna_numero_positivo(resultado: ResultadoTreino):
    valor = prever_valor(resultado.pipeline, 120, 3, 2, "Zona 1")
    assert isinstance(valor, float)
    assert valor > 0


def test_prever_valor_aumenta_com_metragem(resultado: ResultadoTreino):
    valor_pequeno = prever_valor(resultado.pipeline, 60, 2, 1, "Zona 3")
    valor_grande = prever_valor(resultado.pipeline, 250, 2, 1, "Zona 3")
    assert valor_grande > valor_pequeno


def test_salvar_e_carregar_modelo_faz_round_trip(tmp_path, resultado: ResultadoTreino):
    caminho = tmp_path / "modelo_teste.pkl"
    salvar_modelo(resultado.pipeline, str(caminho))

    modelo_carregado = carregar_modelo(str(caminho))
    original = prever_valor(resultado.pipeline, 150, 2, 1, "Zona 5")
    carregado = prever_valor(modelo_carregado, 150, 2, 1, "Zona 5")

    assert original == pytest.approx(carregado)


def test_bairros_disponiveis_sao_conhecidos_pelo_modelo(resultado: ResultadoTreino):
    for bairro in BAIRROS:
        valor = prever_valor(resultado.pipeline, 100, 2, 1, bairro)
        assert valor > 0
