# MLImoveis

Aplicação web para **previsão de preços de imóveis em Maringá (PR)** usando aprendizado de máquina. O usuário informa o tamanho do imóvel (m²) e o bairro e recebe uma estimativa de valor em reais.

## Tecnologias

| Camada | Ferramenta |
|--------|------------|
| Interface web | [Streamlit](https://streamlit.io/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) — Regressão Linear |
| Manipulação de dados | [pandas](https://pandas.pydata.org/) |
| Gerenciamento de dependências | [Poetry](https://python-poetry.org/) |
| Linguagem | Python 3.10+ |

## Pré-requisitos

- Python 3.10 ou superior
- [Poetry](https://python-poetry.org/docs/#installation) instalado

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/dclaumanndeveloper/mlimoveis.git
cd mlimoveis

# 2. Instale as dependências
poetry install

# 3. Ative o ambiente virtual
poetry shell
```

## Como usar

### Executar a aplicação web

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador. Na barra lateral:

1. Informe o **tamanho do imóvel em m²**
2. Selecione o **bairro** (Zona 7 ou Zona 3)
3. Clique em **"Calcular valor do imóvel"**

O valor estimado será exibido na tela principal.

### Retreinar o modelo

Se você atualizar o arquivo `dataset.csv` com novos dados, rode:

```bash
python script.py
```

O script exibirá a acurácia (R²) e atualizará o arquivo `modelo_treinado.pkl`.

## Estrutura do projeto

```
MLImoveis/
├── app.py                  # Interface Streamlit (entrada do usuário → predição)
├── script.py               # Script de treinamento do modelo
├── ml.py                   # Módulo de funções de machine learning
├── dataset.csv             # Dataset de treinamento (100 imóveis)
├── casas.csv               # Dataset auxiliar de exemplo
├── modelo_treinado.pkl     # Modelo serializado (gerado pelo script.py)
├── pyproject.toml          # Configuração de dependências (Poetry)
└── poetry.lock             # Versões travadas das dependências
```

## Arquitetura do modelo

O modelo usa **Regressão Linear** com duas features:

| Feature | Descrição |
|---------|-----------|
| `m2` | Área do imóvel em metros quadrados |
| `bairro` | Bairro codificado numericamente (Zona 3 = 0, Zona 7 = 1) |

**Resultado atual:** R² ≈ 98% com o dataset padrão.

## Limitações e próximos passos

Este projeto é um MVP educacional. Pontos de melhoria para uso em produção:

- [ ] Ampliar o dataset com mais bairros e variáveis (quartos, vagas, ano de construção)
- [ ] Substituir Regressão Linear por modelos mais robustos (Random Forest, Gradient Boosting)
- [ ] Adicionar validação cruzada e análise de resíduos
- [ ] Criar pipeline de retreinamento automatizado
- [ ] Adicionar testes automatizados
- [ ] Expor API REST além da interface Streamlit
