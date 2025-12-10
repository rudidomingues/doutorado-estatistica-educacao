"""
analysis_script.py

Este script em Python realiza uma análise estatística aplicada sobre uma base de dados sintética
que simula escolas com e sem infraestrutura tecnológica. O objetivo é mostrar o uso de
distribuições de probabilidade, medidas descritivas, visualizações e teste de hipóteses
no contexto da disciplina de Estatística Aplicada à Computação.

O script assume que o arquivo CSV `dataset_sintetico.csv` está disponível no mesmo diretório.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def carregar_dados(path: str) -> pd.DataFrame:
    """Carrega a base de dados a partir de um arquivo CSV."""
    return pd.read_csv(path)


def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas da taxa de aprovação por grupo (com/sem tecnologia)."""
    return df.groupby("TEM_ESTRUTURA_TEC")["taxa_aprovacao"].describe()


def graficos_distribuicoes_discretas() -> None:
    """Gera histogramas para distribuições discretas: Bernoulli, Binomial, Geométrica, Poisson e Hipergeométrica."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.ravel()

    # 1. Bernoulli
    bern = np.random.binomial(1, 0.7, size=1000)
    axes[0].hist(bern, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    axes[0].set_title("Bernoulli (p=0,7)")

    # 2. Binomial
    binom = np.random.binomial(10, 0.6, size=4000)
    axes[1].hist(binom, bins=np.arange(0, 12) - 0.5, rwidth=0.8)
    axes[1].set_title("Binomial (n=10, p=0,6)")

    # 3. Geométrica
    geom = np.random.geometric(0.3, size=4000)
    axes[2].hist(geom, bins=np.arange(1, 20), rwidth=0.8)
    axes[2].set_title("Geométrica (p=0,3)")

    # 4. Poisson
    pois = np.random.poisson(4, size=5000)
    axes[3].hist(pois, bins=np.arange(0, 15) - 0.5, rwidth=0.8)
    axes[3].set_title("Poisson (λ=4)")

    # 5. Hipergeométrica
    hiper = np.random.hypergeometric(ngood=40, nbad=110, nsample=20, size=5000)
    axes[4].hist(hiper, bins=np.arange(0, 20) - 0.5, rwidth=0.8)
    axes[4].set_title("Hipergeométrica (M=150, K=40, n=20)")

    # Remove eixo vazio
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig("distribuicoes_discretas.png")
    plt.close(fig)


def graficos_distribuicoes_continuas() -> None:
    """Gera histogramas para distribuições contínuas: Uniforme, Exponencial, Normal e t-Student."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    uni = np.random.uniform(0, 1, 5000)
    axes[0].hist(uni, bins=20)
    axes[0].set_title("Uniforme U(0,1)")

    exp = np.random.exponential(scale=1, size=5000)
    axes[1].hist(exp, bins=30)
    axes[1].set_title("Exponencial (λ=1)")

    norm = np.random.normal(0, 1, 5000)
    axes[2].hist(norm, bins=30)
    axes[2].set_title("Normal N(0,1)")

    t = np.random.standard_t(df=10, size=5000)
    axes[3].hist(t, bins=30)
    axes[3].set_title("t-Student (df=10)")

    plt.tight_layout()
    plt.savefig("distribuicoes_continuas.png")
    plt.close(fig)


def graficos_base_escolas(df: pd.DataFrame) -> None:
    """Gera histogramas, boxplot e dispersão da taxa de aprovação separados por infraestrutura tecnológica."""
    sem_tec = df.loc[df["TEM_ESTRUTURA_TEC"] == 0, "taxa_aprovacao"]
    com_tec = df.loc[df["TEM_ESTRUTURA_TEC"] == 1, "taxa_aprovacao"]

    # Histograma superposto
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(com_tec, bins=15, alpha=0.7, label="Com TEC")
    plt.hist(sem_tec, bins=15, alpha=0.7, label="Sem TEC")
    plt.title("Histograma")
    plt.xlabel("Taxa de aprovação")
    plt.legend()

    # Boxplot
    plt.subplot(1, 3, 2)
    plt.boxplot([sem_tec, com_tec], labels=["Sem TEC", "Com TEC"])
    plt.title("Boxplot")
    plt.ylabel("Taxa de aprovação")

    # Dispersão
    plt.subplot(1, 3, 3)
    plt.scatter(df["TEM_ESTRUTURA_TEC"], df["taxa_aprovacao"])
    plt.title("Dispersão")
    plt.xlabel("Tem tecnologia")
    plt.ylabel("Taxa de aprovação")

    plt.tight_layout()
    plt.savefig("graficos_base_escolas.png")
    plt.close()


def teste_de_hipotese(df: pd.DataFrame) -> tuple:
    """Realiza teste t de Student para a diferença de médias entre escolas com e sem tecnologia.

    Retorna a estatística t e o p-valor.
    """
    sem_tec = df.loc[df["TEM_ESTRUTURA_TEC"] == 0, "taxa_aprovacao"]
    com_tec = df.loc[df["TEM_ESTRUTURA_TEC"] == 1, "taxa_aprovacao"]
    t_stat, p_value = stats.ttest_ind(com_tec, sem_tec, equal_var=False)
    return t_stat, p_value


def main() -> None:
    # Carregar base de dados sintética
    df = carregar_dados("dataset_sintetico.csv")

    # Estatísticas descritivas
    stats_desc = estatisticas_descritivas(df)
    print("Estatísticas descritivas:\n", stats_desc)

    # Geração de gráficos
    graficos_distribuicoes_discretas()
    graficos_distribuicoes_continuas()
    graficos_base_escolas(df)

    # Teste de hipótese
    t_stat, p_val = teste_de_hipotese(df)
    print(f"\nTeste t de Student:\nt = {t_stat:.4f}\np-valor = {p_val:.6f}")
    if p_val < 0.05:
        print("Resultado: Diferença significativa entre os grupos.")
    else:
        print("Resultado: Não há diferença significativa ao nível de 5%.")


if __name__ == "__main__":
    main()