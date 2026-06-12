import streamlit as st
from ml import carregar_modelo, prever_valor

modelo = carregar_modelo('modelo_treinado.pkl')

st.set_page_config(page_title="Previsão de preço de imóveis em Maringá")

st.title("Previsão dos valores de imóveis")
st.caption("Maringá – PR | Modelo de Regressão Linear")
st.divider()

menu = st.sidebar

metragem = menu.number_input("Tamanho do imóvel (m²):", min_value=0.0, step=1.0)
bairro = menu.selectbox(
    label="Bairro",
    options=['Zona 7', 'Zona 3'],
    index=None,
    placeholder="Selecione um bairro",
)
prever_preco = menu.button("Calcular valor do imóvel")

if prever_preco:
    if not metragem:
        st.error("Informe o tamanho do imóvel (m² deve ser maior que 0).")
    elif bairro is None:
        st.error("Selecione um bairro antes de calcular.")
    else:
        valor = prever_valor(modelo, metragem, bairro)
        st.success("Preço previsto com sucesso!")
        st.info(f"O valor estimado do imóvel de {metragem:.0f} m² no {bairro} é de **R$ {valor:,.2f}**.")
