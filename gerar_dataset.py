"""Gera um dataset sintético de imóveis em Maringá (PR).

Não existe, até o momento, um dataset público confiável com preços reais de
imóveis individuais em Maringá (a prefeitura não publica o "valor venal" em
lote, e portais como ZAP/VivaReal/OLX/Imovelweb proíbem coleta automatizada
nos termos de uso). Os preços-base por m² abaixo foram calibrados a partir de
médias de mercado citadas por blogs de imobiliárias locais em 2026 (ex.:
Zona 3 como bairro mais valorizado, ~R$ 6.500–8.500/m²; Zona 8 como região
emergente mais barata, ~R$ 3.800–5.000/m²; média da cidade ~R$ 5.774/m²).
São valores aproximados de conteúdo de marketing, não estatística oficial —
os dados continuam sendo simulados (fórmula + ruído), não transações reais.

Quartos e vagas de garagem somam um valor fixo ao preço final. Um ruído
gaussiano é adicionado para simular variação de mercado. A semente fixa
garante reprodutibilidade.

Uso:
    python gerar_dataset.py
"""
import numpy as np
import pandas as pd

SEED = 42
IMOVEIS_POR_BAIRRO = 60

PRECO_M2_BASE = {
    'Zona 1': 6200,
    'Zona 2': 5200,
    'Zona 3': 7500,
    'Zona 4': 5800,
    'Zona 5': 4700,
    'Zona 6': 6800,
    'Zona 7': 5400,
    'Zona 8': 4400,
}

BONUS_QUARTO = 40000
BONUS_VAGA = 25000
RUIDO_DESVIO_PADRAO = 45000


def gerar_dataset(rng: np.random.Generator) -> pd.DataFrame:
    linhas = []
    for bairro, preco_m2 in PRECO_M2_BASE.items():
        m2 = rng.uniform(40, 220, IMOVEIS_POR_BAIRRO).round(0)
        quartos = rng.integers(1, 5, IMOVEIS_POR_BAIRRO)
        vagas = rng.integers(0, 4, IMOVEIS_POR_BAIRRO)
        ruido = rng.normal(0, RUIDO_DESVIO_PADRAO, IMOVEIS_POR_BAIRRO)

        valor = (m2 * preco_m2) + (quartos * BONUS_QUARTO) + (vagas * BONUS_VAGA) + ruido
        valor = valor.round(2).clip(min=1000)

        for i in range(IMOVEIS_POR_BAIRRO):
            linhas.append({
                'm2': m2[i],
                'quartos': int(quartos[i]),
                'vagas': int(vagas[i]),
                'bairro': bairro,
                'valor': valor[i],
            })

    df = pd.DataFrame(linhas)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    df = gerar_dataset(rng)
    df.to_csv('dataset.csv', index=False)
    print(f"Dataset gerado com {len(df)} imóveis em {len(PRECO_M2_BASE)} bairros -> dataset.csv")
