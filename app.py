import streamlit as st
import pandas as pd
import pickle

#carregando o modelo já treinado
with open("modelo_treinado.pkl","rb") as file:
    modelo = pickle.load(file)
    
    
def calcula_valor(metragem, bairro):    
    dados = pd.DataFrame({'m2': [metragem], 'bairro': [bairro]})
    valor = modelo.predict(dados)[0][0]
    return valor
    
st.set_page_config(
    page_title="Previsão de preço de imóveis em Maringá",
)    

st.title("Previsão dos valores de imóveis")
st.divider()

menu = st.sidebar

metragem = menu.number_input("Digite o tamanho do imóvel (m2):")
bairro = menu.selectbox(label="Bairro",options=['Zona 7','Zona 3'],index=None,placeholder="Selecione um bairro")
prever_preco = menu.button("Calcular valor do imóvel")

if prever_preco:
    if not metragem :
        st.error("O valor do imóvel não pode ser R$0.")
    
    else:
        valor = calcula_valor(metragem, bairro)
        #Mensagens de sucesso e previsão
        st.success("Preço previsto com sucesso!")
        st.info(f"O valor do imóvel de {metragem:.2f} m2 no bairro {bairro} é de R$ {valor:,.2f}.")
        
        

