# Predição de Atropelamentos de Fauna Silvestre – RodoFauna (IBRAM/DF)

Trabalho de Aprendizado Supervisionado – Inteligência Artificial (UFSM).
Grupo: Gabriel Vargas Saueressig e Sofia Valentina.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ordem de execução (sempre a partir da raiz do repositório)

| Etapa | Comando | Saídas |
| :- | :- | :- |
| 1. Análise exploratória | `python analise_dados.py` | gráficos na tela |
| 2. Pré-processamento (multiclasse) | `python etapa2/pre_processamento_multi_classe.py` | `dataSet/treinoTesteValidacao/*.csv` |
| 2. Pré-processamento (segmentos 500 m) | `python etapa2/pre_processamento_binario_regressao.py` | `dataSet/treinoTesteValidacaoSegmentos/*.csv` |
| 3. Classificação multiclasse | `python etapa3/classificacao_multiclasse.py` | `img/matriz_confusao_*.png`, `img/curvas_treinamento_mlp.png` |
| 3. Classificação binária | `python etapa3/classificacao_binaria.py` | `img/curva_roc_binaria.png`, `img/matriz_confusao_binaria.png` |
| 4. Regressão | `python etapa4/regressao.py` | `img/curva_loss_regressao.png`, `img/reais_vs_preditos_mlp.png`, `img/grafico_residuos_mlp.png` |
| 5. Optuna (~10 min) | `python etapa5/otimizacao_optuna.py` | `resultados/optuna_*`, `resultados/etapa5_*`, `img/optuna_*.png` |
| 6. Regularização (~15–20 min) | `python etapa6/regularizacao_overfitting.py` | `resultados/etapa6_regularizacao.csv`, `img/etapa6_*.png` |
| 7. Análises da discussão | `python etapa7/analises_discussao.py` | `resultados/etapa7_*`, `img/etapa7_*.png` |

As Etapas 6 e 7 leem os hiperparâmetros encontrados na Etapa 5 (`resultados/etapa5_resumo.json`).
Os trials do Optuna ficam salvos em `resultados/optuna_trials.csv`.

Todas as sementes aleatórias são fixadas em 42.
