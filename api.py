from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml import BAIRROS, carregar_modelo, prever_valor

app = FastAPI(
    title="MLImoveis API",
    description="Previsão de preços de imóveis em Maringá (PR)",
    version="1.0.0",
)

modelo = carregar_modelo("modelo_treinado.pkl")


class ImovelEntrada(BaseModel):
    m2: float = Field(gt=0, description="Tamanho do imóvel em metros quadrados")
    quartos: int = Field(ge=1, description="Número de quartos")
    vagas: int = Field(ge=0, description="Número de vagas de garagem")
    bairro: str = Field(description="Bairro do imóvel")


class PrecoSaida(BaseModel):
    valor_previsto: float


@app.get("/")
def raiz():
    return {"status": "ok", "servico": "MLImoveis API"}


@app.get("/bairros", response_model=list[str])
def listar_bairros():
    return BAIRROS


@app.post("/prever", response_model=PrecoSaida)
def prever(imovel: ImovelEntrada):
    if imovel.bairro not in BAIRROS:
        raise HTTPException(
            status_code=422,
            detail=f"Bairro inválido. Opções válidas: {', '.join(BAIRROS)}",
        )

    valor = prever_valor(modelo, imovel.m2, imovel.quartos, imovel.vagas, imovel.bairro)
    return PrecoSaida(valor_previsto=round(valor, 2))
