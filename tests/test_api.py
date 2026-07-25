from fastapi.testclient import TestClient

from api import app

cliente = TestClient(app)


def test_raiz_responde_ok():
    resposta = cliente.get("/")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_listar_bairros_retorna_lista_nao_vazia():
    resposta = cliente.get("/bairros")
    assert resposta.status_code == 200
    assert len(resposta.json()) > 0


def test_prever_com_dados_validos():
    resposta = cliente.post(
        "/prever",
        json={"m2": 120, "quartos": 3, "vagas": 2, "bairro": "Zona 1"},
    )
    assert resposta.status_code == 200
    assert resposta.json()["valor_previsto"] > 0


def test_prever_com_bairro_invalido_retorna_422():
    resposta = cliente.post(
        "/prever",
        json={"m2": 120, "quartos": 3, "vagas": 2, "bairro": "Bairro Inexistente"},
    )
    assert resposta.status_code == 422


def test_prever_com_m2_negativo_retorna_422():
    resposta = cliente.post(
        "/prever",
        json={"m2": -10, "quartos": 3, "vagas": 2, "bairro": "Zona 1"},
    )
    assert resposta.status_code == 422
