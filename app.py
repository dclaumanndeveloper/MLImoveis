import datetime

import streamlit as st
from ml import BAIRROS, TIPOS_IMOVEL, carregar_modelo, prever_valor

modelo = carregar_modelo('modelo_treinado.pkl')

st.set_page_config(page_title="Previsão de preço de imóveis em Maringá")

st.title("Previsão dos valores de imóveis")
st.caption("Maringá – PR | Modelo de Machine Learning")
st.divider()

menu = st.sidebar
ano_atual = datetime.date.today().year

metragem = menu.number_input("Tamanho do imóvel (m²):", min_value=0.0, step=1.0)
quartos = menu.number_input("Quartos:", min_value=1, max_value=10, value=1, step=1)
vagas = menu.number_input("Vagas de garagem:", min_value=0, max_value=10, value=0, step=1)
tipo_imovel = menu.selectbox("Tipo de imóvel", options=TIPOS_IMOVEL, index=None, placeholder="Selecione o tipo")
ano_construcao = menu.number_input(
    "Ano de construção:", min_value=1900, max_value=ano_atual, value=2010, step=1
)
distancia_centro_km = menu.number_input(
    "Distância do centro (km):", min_value=0.0, max_value=50.0, value=5.0, step=0.5
)
bairro = menu.selectbox(
    label="Bairro",
    options=BAIRROS,
    index=None,
    placeholder="Selecione um bairro",
)
prever_preco = menu.button("Calcular valor do imóvel")

if prever_preco:
    if not metragem:
        st.error("Informe o tamanho do imóvel (m² deve ser maior que 0).")
    elif bairro is None:
        st.error("Selecione um bairro antes de calcular.")
    elif tipo_imovel is None:
        st.error("Selecione o tipo de imóvel antes de calcular.")
    else:
        valor = prever_valor(
            modelo, metragem, quartos, vagas, ano_construcao, distancia_centro_km, bairro, tipo_imovel
        )
        st.success("Preço previsto com sucesso!")
        st.info(
            f"O valor estimado do(a) {tipo_imovel.lower()} de {metragem:.0f} m², {quartos} quarto(s), "
            f"{vagas} vaga(s), construído(a) em {ano_construcao} e a {distancia_centro_km:.1f} km do "
            f"centro, no {bairro}, é de **R$ {valor:,.2f}**."
        )
