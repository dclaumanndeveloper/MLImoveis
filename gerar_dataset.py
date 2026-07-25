"""Gera um dataset sintético de imóveis em Maringá (PR).

Cada bairro tem um preço-base por m² distinto; quartos e vagas de garagem
somam um valor fixo ao preço final. Um ruído gaussiano é adicionado para
simular variação de mercado. A semente fixa garante reprodutibilidade.

Uso:
    python gerar_dataset.py
"""
import numpy as np
import pandas as pd

SEED = 42
IMOVEIS_POR_BAIRRO = 60

PRECO_M2_BASE = {
    'Zona 1': 1200,
    'Zona 2': 950,
    'Zona 3': 850,
    'Zona 4': 1100,
    'Zona 5': 780,
    'Zona 6': 1300,
    'Zona 7': 900,
    'Zona 8': 1050,
}

BONUS_QUARTO = 8000
BONUS_VAGA = 5000
RUIDO_DESVIO_PADRAO = 9000


def gerar_dataset(rng: np.random.Generator) -> pd.DataFrame:
    linhas = []
    for bairro, preco_m2 in PRECO_M2_BASE.items():
        m2 = rng.uniform(40, 300, IMOVEIS_POR_BAIRRO).round(0)
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
