import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml import BAIRROS, TIPOS_IMOVEL, carregar_modelo, prever_valor

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
    ano_construcao: int = Field(
        ge=1900, le=datetime.date.today().year, description="Ano de construção do imóvel"
    )
    distancia_centro_km: float = Field(ge=0, description="Distância do centro da cidade em km")
    bairro: str = Field(description="Bairro do imóvel")
    tipo_imovel: str = Field(description="Tipo do imóvel (Casa ou Apartamento)")


class PrecoSaida(BaseModel):
    valor_previsto: float


@app.get("/")
def raiz():
    return {"status": "ok", "servico": "MLImoveis API"}


@app.get("/bairros", response_model=list[str])
def listar_bairros():
    return BAIRROS


@app.get("/tipos-imovel", response_model=list[str])
def listar_tipos_imovel():
    return TIPOS_IMOVEL


@app.post("/prever", response_model=PrecoSaida)
def prever(imovel: ImovelEntrada):
    if imovel.bairro not in BAIRROS:
        raise HTTPException(
            status_code=422,
            detail=f"Bairro inválido. Opções válidas: {', '.join(BAIRROS)}",
        )

    if imovel.tipo_imovel not in TIPOS_IMOVEL:
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de imóvel inválido. Opções válidas: {', '.join(TIPOS_IMOVEL)}",
        )

    valor = prever_valor(
        modelo,
        imovel.m2,
        imovel.quartos,
        imovel.vagas,
        imovel.ano_construcao,
        imovel.distancia_centro_km,
        imovel.bairro,
        imovel.tipo_imovel,
    )
    return PrecoSaida(valor_previsto=round(valor, 2))
