import os

import pandas as pd
import pytest

from ml import (
    BAIRROS,
    TIPOS_IMOVEL,
    ResultadoTreino,
    carregar_modelo,
    prever_valor,
    preparar_dados,
    salvar_modelo,
    treinar_modelo,
    verificar_saude_modelo,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    caminho = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")
    return pd.read_csv(caminho)


@pytest.fixture(scope="module")
def resultado(dataset: pd.DataFrame) -> ResultadoTreino:
    return treinar_modelo(dataset)


def prever(pipeline, m2=120, quartos=3, vagas=2, ano_construcao=2015, distancia_centro_km=5.0,
           bairro="Zona 1", tipo_imovel="Casa"):
    return prever_valor(pipeline, m2, quartos, vagas, ano_construcao, distancia_centro_km, bairro, tipo_imovel)


def test_treinar_modelo_retorna_resultado_valido(resultado: ResultadoTreino):
    assert resultado.pipeline is not None
    assert resultado.nome_modelo in {"Regressão Linear", "Random Forest", "Gradient Boosting"}
    assert resultado.r2_teste > 80
    assert resultado.mae_teste > 0
    assert resultado.rmse_teste > 0
    assert set(resultado.comparacao) == {"Regressão Linear", "Random Forest", "Gradient Boosting"}


def test_preparar_dados_gera_colunas_esperadas():
    df = preparar_dados(120, 3, 2, 2015, 5.0, "Zona 1", "Casa")
    assert list(df.columns) == [
        "m2", "quartos", "vagas", "ano_construcao", "distancia_centro_km", "bairro", "tipo_imovel",
    ]
    assert df.shape == (1, 7)
    assert df.iloc[0]["bairro"] == "Zona 1"
    assert df.iloc[0]["tipo_imovel"] == "Casa"


def test_prever_valor_retorna_numero_positivo(resultado: ResultadoTreino):
    valor = prever(resultado.pipeline)
    assert isinstance(valor, float)
    assert valor > 0


def test_prever_valor_aumenta_com_metragem(resultado: ResultadoTreino):
    valor_pequeno = prever(resultado.pipeline, m2=60, bairro="Zona 3")
    valor_grande = prever(resultado.pipeline, m2=250, bairro="Zona 3")
    assert valor_grande > valor_pequeno


def test_prever_valor_diminui_com_distancia_do_centro(resultado: ResultadoTreino):
    valor_perto = prever(resultado.pipeline, distancia_centro_km=1.0)
    valor_longe = prever(resultado.pipeline, distancia_centro_km=15.0)
    assert valor_perto > valor_longe


def test_salvar_e_carregar_modelo_faz_round_trip(tmp_path, resultado: ResultadoTreino):
    caminho = tmp_path / "modelo_teste.pkl"
    salvar_modelo(resultado.pipeline, str(caminho))

    modelo_carregado = carregar_modelo(str(caminho))
    original = prever(resultado.pipeline, bairro="Zona 5")
    carregado = prever(modelo_carregado, bairro="Zona 5")

    assert original == pytest.approx(carregado)


def test_bairros_disponiveis_sao_conhecidos_pelo_modelo(resultado: ResultadoTreino):
    for bairro in BAIRROS:
        valor = prever(resultado.pipeline, bairro=bairro)
        assert valor > 0


def test_tipos_imovel_disponiveis_sao_conhecidos_pelo_modelo(resultado: ResultadoTreino):
    for tipo_imovel in TIPOS_IMOVEL:
        valor = prever(resultado.pipeline, tipo_imovel=tipo_imovel)
        assert valor > 0


def test_verificar_saude_modelo_detecta_r2_abaixo_do_minimo():
    problemas = verificar_saude_modelo(70.0, None)
    assert len(problemas) == 1


def test_verificar_saude_modelo_detecta_queda_abrupta():
    problemas = verificar_saude_modelo(88.0, 96.0)
    assert len(problemas) == 1


def test_verificar_saude_modelo_sem_problemas():
    assert verificar_saude_modelo(96.0, 95.0) == []
    assert verificar_saude_modelo(96.0, None) == []
