from fastapi.testclient import TestClient

from api import app

cliente = TestClient(app)

IMOVEL_VALIDO = {
    "m2": 120,
    "quartos": 3,
    "vagas": 2,
    "ano_construcao": 2015,
    "distancia_centro_km": 5.0,
    "bairro": "Zona 1",
    "tipo_imovel": "Casa",
}


def test_raiz_responde_ok():
    resposta = cliente.get("/")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_listar_bairros_retorna_lista_nao_vazia():
    resposta = cliente.get("/bairros")
    assert resposta.status_code == 200
    assert len(resposta.json()) > 0


def test_listar_tipos_imovel_retorna_lista_nao_vazia():
    resposta = cliente.get("/tipos-imovel")
    assert resposta.status_code == 200
    assert len(resposta.json()) > 0


def test_prever_com_dados_validos():
    resposta = cliente.post("/prever", json=IMOVEL_VALIDO)
    assert resposta.status_code == 200
    assert resposta.json()["valor_previsto"] > 0


def test_prever_com_bairro_invalido_retorna_422():
    dados = {**IMOVEL_VALIDO, "bairro": "Bairro Inexistente"}
    resposta = cliente.post("/prever", json=dados)
    assert resposta.status_code == 422


def test_prever_com_tipo_imovel_invalido_retorna_422():
    dados = {**IMOVEL_VALIDO, "tipo_imovel": "Chácara"}
    resposta = cliente.post("/prever", json=dados)
    assert resposta.status_code == 422


def test_prever_com_m2_negativo_retorna_422():
    dados = {**IMOVEL_VALIDO, "m2": -10}
    resposta = cliente.post("/prever", json=dados)
    assert resposta.status_code == 422


def test_prever_com_ano_construcao_futuro_retorna_422():
    dados = {**IMOVEL_VALIDO, "ano_construcao": 3000}
    resposta = cliente.post("/prever", json=dados)
    assert resposta.status_code == 422
