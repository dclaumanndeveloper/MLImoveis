# MLImoveis

Aplicação web para **previsão de preços de imóveis em Maringá (PR)** usando aprendizado de máquina. O usuário informa o tamanho do imóvel (m²), quartos, vagas de garagem, tipo de imóvel, ano de construção, distância do centro e o bairro, e recebe uma estimativa de valor em reais.

## Tecnologias

| Camada | Ferramenta |
|--------|------------|
| Interface web | [Streamlit](https://streamlit.io/) |
| API REST | [FastAPI](https://fastapi.tiangolo.com/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) — Regressão Linear, Random Forest e Gradient Boosting |
| Manipulação de dados | [pandas](https://pandas.pydata.org/) |
| Testes | [pytest](https://docs.pytest.org/) |
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
2. Informe o **número de quartos** e **vagas de garagem**
3. Selecione o **tipo de imóvel** (Casa ou Apartamento)
4. Informe o **ano de construção** e a **distância do centro (km)**
5. Selecione o **bairro**
6. Clique em **"Calcular valor do imóvel"**

O valor estimado será exibido na tela principal.

### Executar a API REST

```bash
uvicorn api:app --reload
```

A documentação interativa fica disponível em `http://localhost:8000/docs`.

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Health check |
| `/bairros` | GET | Lista os bairros suportados pelo modelo |
| `/tipos-imovel` | GET | Lista os tipos de imóvel suportados (Casa, Apartamento) |
| `/prever` | POST | Recebe `{"m2", "quartos", "vagas", "ano_construcao", "distancia_centro_km", "bairro", "tipo_imovel"}` e retorna `{"valor_previsto"}` |

Exemplo:

```bash
curl -X POST http://localhost:8000/prever \
  -H "Content-Type: application/json" \
  -d '{"m2": 120, "quartos": 3, "vagas": 2, "ano_construcao": 2015, "distancia_centro_km": 5.0, "bairro": "Zona 1", "tipo_imovel": "Casa"}'
```

### Gerar/atualizar o dataset

O dataset é **sintético** (fórmula + ruído, semente fixa para reprodutibilidade) — não existe hoje um dataset público confiável com preços reais de imóveis individuais em Maringá: a prefeitura não publica o "valor venal" em lote (só consulta por imóvel específico) e os portais de anúncios (ZAP, VivaReal, OLX, Imovelweb) proíbem coleta automatizada nos termos de uso.

O preço-base por m² de cada bairro foi calibrado com médias de mercado citadas por blogs de imobiliárias locais em 2026 (ex.: Zona 3 como bairro mais valorizado, ~R$ 6.500–8.500/m²; Zona 8 como região emergente mais barata, ~R$ 3.800–5.000/m²; média da cidade ~R$ 5.774/m²) — valores de conteúdo de marketing, não estatística oficial, então servem apenas para dar uma ordem de grandeza e uma hierarquia plausível entre bairros. Quartos, vagas, ano de construção e distância do centro somam/subtraem um valor, apartamentos recebem um pequeno prêmio sobre casas, e um ruído gaussiano simula variação de mercado:

```bash
python gerar_dataset.py
```

### Retreinar o modelo

Sempre que o `dataset.csv` for atualizado, rode:

```bash
python script.py
```

O script compara três modelos (Regressão Linear, Random Forest e Gradient Boosting) via validação cruzada (5 folds), escolhe automaticamente o de melhor R² médio, avalia-o em um conjunto de teste isolado (R², MAE, RMSE) e salva o vencedor em `modelo_treinado.pkl`.

Cada execução também registra as métricas em `metrics_historico.csv` (uma linha por treino, com timestamp) e roda uma checagem de saúde/drift (`verificar_saude_modelo` em `ml.py`): o script falha (`exit 1`) e **não** sobrescreve `modelo_treinado.pkl` se o R² do teste cair abaixo de 85% ou cair mais de 5 pontos em relação ao treino anterior — sinal de que os dados/modelo pioraram e merecem investigação antes de ir para produção.

### Retreinamento agendado

O workflow `.github/workflows/retrain.yml` roda automaticamente toda segunda-feira (e também pode ser disparado manualmente via `workflow_dispatch`): reinstala as dependências, executa `script.py` e, se a checagem de saúde/drift passar, faz commit automático do `modelo_treinado.pkl` e do `metrics_historico.csv` atualizados direto no `main`. Se a checagem falhar, o job falha e nada é commitado — o modelo em produção continua sendo o último saudável.

### Rodar os testes automatizados

```bash
pytest
```

Os testes cobrem o treinamento/seleção de modelo, o pré-processamento de dados de entrada, a predição, a serialização do modelo e os endpoints da API REST.

## Integração contínua (CI)

O workflow `.github/workflows/ci.yml` roda automaticamente em pushes para `main` e em pull requests: instala as dependências via Poetry, executa a suíte `pytest` e treina o modelo (`script.py`) como smoke test do pipeline de ML. Já o `.github/workflows/retrain.yml` cuida do retreinamento agendado (veja a seção acima).

## Estrutura do projeto

```
MLImoveis/
├── app.py                     # Interface Streamlit (entrada do usuário → predição)
├── api.py                     # API REST (FastAPI)
├── script.py                  # Treina, compara modelos, checa saúde/drift e salva o melhor
├── ml.py                      # Pipeline de pré-processamento, treino, predição e checagem de saúde
├── gerar_dataset.py           # Geração reprodutível do dataset sintético
├── dataset.csv                # Dataset de treinamento (480 imóveis, 8 bairros)
├── metrics_historico.csv      # Histórico de métricas de cada treino (gerado pelo script.py)
├── casas.csv                  # Dataset auxiliar de exemplo
├── modelo_treinado.pkl        # Pipeline serializado (gerado pelo script.py)
├── tests/                     # Testes automatizados (pytest)
│   ├── test_ml.py
│   └── test_api.py
├── .github/workflows/
│   ├── ci.yml                 # Testes a cada push/PR
│   └── retrain.yml            # Retreinamento agendado com checagem de drift
├── pyproject.toml             # Configuração de dependências (Poetry)
└── poetry.lock                 # Versões travadas das dependências
```

## Arquitetura do modelo

O `ml.py` monta um `Pipeline` scikit-learn com:

1. **Pré-processamento**: `OneHotEncoder` para as features categóricas (bairro, tipo de imóvel) + passthrough para as features numéricas.
2. **Seleção de modelo**: `treinar_modelo` treina e compara três algoritmos via validação cruzada (5 folds) e escolhe automaticamente o de melhor R² médio.
3. **Checagem de saúde/drift**: `verificar_saude_modelo` sinaliza se o R² do teste está abaixo de um piso mínimo ou caiu abruptamente em relação ao treino anterior (usado por `script.py` e pelo retreinamento agendado).

| Feature | Descrição |
|---------|-----------|
| `m2` | Área do imóvel em metros quadrados |
| `quartos` | Número de quartos |
| `vagas` | Número de vagas de garagem |
| `ano_construcao` | Ano de construção do imóvel |
| `distancia_centro_km` | Distância do centro da cidade, em km |
| `bairro` | Bairro (codificado via one-hot encoding) |
| `tipo_imovel` | Casa ou Apartamento (codificado via one-hot encoding) |

**Resultado atual:** R² ≈ 96% no conjunto de teste com o dataset padrão (Gradient Boosting escolhido automaticamente), MAE ≈ R$ 54,7 mil e RMSE ≈ R$ 69,4 mil — números altos em reais porque o dataset simula preços de imóvel em escala real (centenas de milhares de reais), calibrados como descrito acima.

## Limitações e próximos passos

Este projeto é um MVP educacional. Pontos já endereçados e limitação restante:

- [x] Ampliar o dataset com mais bairros e variáveis (quartos, vagas)
- [x] Substituir Regressão Linear por modelos mais robustos (Random Forest, Gradient Boosting) com seleção automática
- [x] Adicionar validação cruzada e métricas de erro (R², MAE, RMSE)
- [x] Adicionar testes automatizados
- [x] Expor API REST além da interface Streamlit
- [x] Configurar CI (GitHub Actions) rodando os testes a cada push/PR
- [x] Adicionar mais variáveis (ano de construção, distância do centro, tipo de imóvel)
- [x] Agendar retreinamento periódico do modelo no CI, com monitoramento de drift (R² mínimo + queda abrupta)
- [ ] Substituir o dataset sintético por dados reais de mercado

O último item é a única limitação estrutural que resta: não existe hoje uma fonte pública e legítima de preços de imóveis individuais em Maringá (pesquisado em profundidade — ver seção "Gerar/atualizar o dataset"). Virar esse item depende de uma fonte de dados real: uma parceria/API comercial com um portal de anúncios, um pedido de acesso à informação à prefeitura pelo "valor venal" (dado fiscal, não preço de mercado), ou dados fornecidos manualmente por quem tiver acesso a anúncios reais.
