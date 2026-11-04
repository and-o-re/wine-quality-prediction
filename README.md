# 🍷 Classificando a Qualidade de Vinhos com Machine Learning

**POS TECH — Data Analytics (DTAT) | Tech Challenge - Fase 2**

## 📌 1. Contexto e Compreensão do Problema

A indústria vitivinícola possui grande relevância econômica e cultural em diversos países. Tradicionalmente, a avaliação da qualidade de um vinho é realizada por especialistas por meio de análises sensoriais (avaliando aroma, sabor, acidez e equilíbrio). Contudo, esse processo pode ser subjetivo, demandar bastante tempo e depender fortemente da experiência dos avaliadores.

Com o avanço das técnicas de Ciência de Dados e Aprendizado de Máquina, tornou-se possível utilizar dados físico-químicos obtidos durante a produção para auxiliar na previsão da qualidade final da bebida. Modelos preditivos apoiam enólogos e produtores na tomada de decisão, permitindo ajustes rápidos no processo produtivo e contribuindo para a padronização da qualidade.

### 🎯 Objetivo do Projeto

Desenvolver uma pipeline completa de análise exploratória, pré-processamento sem vazamento de dados (*data leakage*) e modelagem preditiva capaz de classificar a qualidade de um vinho com base em suas propriedades físico-químicas.

Para fins de modelagem, a nota de qualidade original (`quality`, situada no intervalo de $3$ a $8$ nesta base) foi transformada em uma **variável alvo binária** (`quality_class`):

* **Vinho de Alta Qualidade (`1` — Premium):** $\text{quality} \ge 7$ — **159 amostras (13,91%)**

* **Vinho de Baixa/Média Qualidade (`0` — Comum):** $\text{quality} < 7$ — **984 amostras (86,09%)**

## 📊 2. Sobre o Conjunto de Dados (Dataset)

O projeto utiliza a base pública **Wine Quality Dataset** (`WineQT.csv`, disponível no [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)), composta por **1.143 observações** e **13 colunas** (11 variáveis preditoras físico-químicas, 1 variável alvo de qualidade e 1 identificador único `Id`). Não há valores ausentes na base.

| Variável | Descrição | Média | Mediana | Mínimo | Máximo | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| `fixed acidity` | Acidez fixa (ácidos não voláteis, como o tartárico) | $8,311$ | $7,900$ | $4,600$ | $15,900$ | 
| `volatile acidity` | Acidez volátil (ácido acético; níveis altos geram sabor avinagrado) | $0,531$ | $0,520$ | $0,120$ | $1,580$ | 
| `citric acid` | Ácido cítrico (adiciona frescor e sabor aos vinhos) | $0,268$ | $0,250$ | $0,000$ | $1,000$ | 
| `residual sugar` | Açúcar residual restante após o término da fermentação | $2,532$ | $2,200$ | $0,900$ | $15,500$ | 
| `chlorides` | Cloretos (quantidade de sal presente no vinho) | $0,087$ | $0,079$ | $0,012$ | $0,611$ | 
| `free sulfur dioxide` | Dióxido de enxofre livre ($SO_2$ livre; previne oxidação e bactérias) | $15,615$ | $13,000$ | $1,000$ | $68,000$ | 
| `total sulfur dioxide` | Dióxido de enxofre total (soma das formas livre e ligada de $SO_2$) | $45,915$ | $37,000$ | $6,000$ | $289,000$ | 
| `density` | Densidade do vinho (varia conforme álcool e açúcar) | $0,997$ | $0,997$ | $0,990$ | $1,004$ | 
| `pH` | Escala de acidez ou basicidade do vinho | $3,311$ | $3,310$ | $2,740$ | $4,010$ | 
| `sulphates` | Sulfatos (aditivo antimicrobiano e antioxidante) | $0,658$ | $0,620$ | $0,330$ | $2,000$ | 
| `alcohol` | Teor alcoólico percentual por volume | $10,442$ | $10,200$ | $8,400$ | $14,900$ | 
| `quality` | **Variável Alvo Original:** Nota sensorial atribuída por especialistas | $5,657$ | $6,000$ | $3,000$ | $8,000$ | 

## 🛠️ 3. Pipeline Metodológica do Desafio

O desenvolvimento deste trabalho foi estruturado em notebooks sequenciais contemplando as 6 etapas exigidas:

1. **Compreensão do Problema e Inspeção (`01_eda.ipynb`):**

   * Verificação de integridade (0 valores nulos) e identificação de **125 registros com perfis físico-químicos idênticos** (distribuídos em 114 grupos, sem divergência de classe).

   * Binarização da variável alvo ($\text{quality} \ge 7$), evidenciando forte desbalanceamento (82,68% das amostras originais concentram-se nas notas 5 e 6, e apenas 13,91% compõem a classe positiva).

2. **Análise Exploratória de Dados (`01_eda.ipynb`):**

   * Avaliação de assimetria positiva acentuada em variáveis como `residual sugar`, `chlorides` e `total sulfur dioxide`.

   * Quantificação de *outliers* pelo critério de $1,5 \times \text{IQR}$: **295 amostras (25,81%)** apresentaram valores extremos em ao menos uma variável (com destaque para `residual sugar` com 9,62% e `chlorides` com 6,74%). As amostras foram preservadas por passarem nos testes de consistência físico-química.

   * Análise de correlação de Pearson com a classe de alta qualidade (`quality_class`).

3. **Pré-processamento e Prevenção de Vazamento (`02_preprocessamento.ipynb`):**

   * **Divisão Estratificada por Grupos (`StratifiedGroupKFold`):** Separação em **914 amostras de treino (79,97%)** e **229 amostras de teste (20,03%)**, mantendo a proporção de vinhos de alta qualidade (~13,9%) e garantindo **0 grupos de perfis repetidos compartilhados** entre treino e teste.

   * **Padronização (`StandardScaler`):** Ajustada exclusivamente no conjunto de treino e exportada em artefato (`standard_scaler.joblib`), além de embutida em `Pipeline` durante a validação cruzada.

   * **Feature Engineering Experimental:** Criação e avaliação no treino da variável `free_so2_ratio` ($\text{free sulfur dioxide} / \text{total sulfur dioxide}$), que apresentou mediana de $43,33\%$ na classe Alta contra $35,71\%$ na classe Baixa/Média ($r = 0,1325$).

4. **Desenvolvimento e Validação de Modelos (`03_modelo.ipynb` / `03_modelo_v2.ipynb`):**

   * Avaliação de algoritmos em validação cruzada (`StratifiedGroupKFold` em 5 dobras no treino): **Regressão Logística**, **Random Forest**, **KNN** e **XGBoost**.

5. **Otimização e Ajuste de Limiar (`03_modelo.ipynb` / `03_modelo_v2.ipynb`):**

   * Refinamento de hiperparâmetros do modelo campeão (**Random Forest**) via `GridSearchCV` focado em F1-Score.

   * Otimização do limiar de decisão (*threshold tuning*) via Curva Precision-Recall para maximizar a eficiência operacional na detecção de vinhos premium.

## 📈 4. Principais Resultados e Comparativo de Modelos

### 4.1. Comparativo de Algoritmos (Validação Cruzada 5-Folds no Treino)

Durante a fase de seleção de modelos no conjunto de treinamento (`X_train`), utilizando `Pipeline` com `StandardScaler` e isolamento de grupos idênticos, obtivemos as seguintes métricas médias para a classe positiva (Alta Qualidade):

| Modelo | Precision Média | Recall Médio | F1-Score Médio | ROC-AUC Média | Comportamento Observado | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Regressão Logística** *(Baseline)* | `0.3655` | `0.7785` | `0.4958` | `0.8622` | Alto Recall, mas excesso de falsos positivos (baixa precisão). | 
| **Random Forest** *(Padrão)* | **`0.7637`** | `0.3142` | `0.4409` | **`0.8678`** | Maior precisão e melhor separação (ROC-AUC), mas conservador no limiar padrão de 50%. | 
| **KNN** *(K=5, distance)* | `0.5002` | `0.3914` | `0.4341` | `0.7582` | Sofreu severamente com o desbalanceamento espacial das classes. | 
| **XGBoost** *(scale_pos_weight)* | `0.5324` | `0.4480` | `0.4798` | `0.8596` | Desempenho intermediário; boa separação hierárquica, mas superado pelo RF após ajuste. | 

> **Escolha do Modelo:** O **Random Forest** foi selecionado como modelo principal por apresentar a maior capacidade de distinção global (**ROC-AUC de 86,78%**) e altíssima confiabilidade inicial (**Precision de 76,37%**), permitindo expandir sua sensibilidade (Recall) por meio de poda de árvores e ajuste do limiar de probabilidade.

### 4.2. Otimização de Hiperparâmetros e Ajuste de Limiar (Conjunto de Teste)

Com o `GridSearchCV`, a melhor arquitetura encontrada para o Random Forest foi:

* `max_depth`: `10` | `min_samples_leaf`: `4` | `min_samples_split`: `10` | `n_estimators`: `100` | `class_weight`: `'balanced_subsample'`

* **F1-Score na Validação Cruzada (Treino):** subiu de `0.4409` para **`0.5555`**.

Ao avaliar o modelo otimizado no **conjunto de teste isolado (229 amostras, sendo 197 comuns e 32 premium)**, comparamos o limiar padrão ($50\%$) com os limiares otimizados via Curva Precision-Recall:

| Cenário no Conjunto de Teste (`X_test`) | Acurácia Global | Precision (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) | Matriz de Confusão `[VN, FP] / [FN, VP]` | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **1. RF Otimizado (Limiar Padrão `50,0%`)** | `0.87` (87%) | `0.53` (53%) | `0.62` (62%) | `0.57` | `[179, 18]` / `[12, 20]` | 
| **2. RF Otimizado (Limiar CV Treino `45,6%`)** | `0.87` (87%) | `0.52` (52%) | `0.69` (69%) | `0.59` | `[177, 20]` / `[10, 22]` | 
| **3. RF Otimizado (Limiar Pico PR `43,0%`)** | **`0.88` (88%)** | **`0.55` (55%)** | **`0.88` (88%)** | **`0.67`** | **`[174, 23]` / `[4, 28]`** | 

* **Impacto Prático no Negócio:** Reduzindo a régua de decisão de $50\%$ para a faixa entre $43,0\%$ **e** $45,6\%$, o Recall no conjunto de teste salta de **62% para até 88%** (capturando até 28 das 32 garrafas premium do lote de teste), elevando o **F1-Score para até 0,67** e mantendo a **Acurácia global em 87%–88%**.

### 4.3. Variáveis de Maior Influência e Implicações na Produção

A análise estatística e de medianas por classe demonstrou que nenhuma variável isolada separa perfeitamente os vinhos, mas 4 fatores químicos lideram a influência na qualidade:

| Variável Físico-Química | Correlação ($r$) | Mediana (Baixa/Média `< 7`) | Mediana (Alta `≥ 7`) | Implicação para o Processo Produtivo | 
| ----- | ----- | ----- | ----- | ----- | 
| **Teor Alcoólico (`alcohol`)** | **`+0,404`** | `10,00%` | **`11,60%`** | Principal indicador positivo; reflete maturação adequada da uva e eficiência na fermentação. | 
| **Acidez Volátil (`volatile acidity`)** | **`-0,305`** | `0,55 g/dm³` | **`0,36 g/dm³`** | Principal indicador negativo; exige rigoroso controle sanitário e de oxigenação para evitar formação de ácido acético (vinagre). | 
| **Ácido Cítrico (`citric acid`)** | **`+0,251`** | `0,24 g/dm³` | **`0,41 g/dm³`** | Contribui para o frescor e equilíbrio ácido dos lotes de alta qualidade. | 
| **Sulfatos (`sulphates`)** | **`+0,208`** | `0,60 g/dm³` | **`0,74 g/dm³`** | Associado à estabilidade antioxidante e proteção microbiológica durante a maturação. | 

## 📁 5. Estrutura do Repositório

```
wine-quality-prediction/
│
├── data/
│   ├── raw/
│   │   └── WineQT.csv                      # Base de dados original imutável (1.143 amostras)
│   └── processed/
│       └── divisao_treino_teste.csv        # Registro determinístico da divisão estratificada por grupos
│
├── notebooks/
│   ├── 01_eda.ipynb                        # Inspeção, estatísticas, outliers (IQR), consistência e correlações
│   ├── 02_preprocessamento.ipynb           # Auditoria de duplicatas, StratifiedGroupKFold, StandardScaler e Feature Engineering
│   ├── 03_modelo.ipynb                     # Modelagem (LR, RF, KNN), GridSearchCV, Threshold Tuning e Inferência
│   └── 03_modelo_v2.ipynb                  # Versão estendida incluindo avaliação com XGBoost e cross_val_predict
│
├── results/
│   ├── artifacts/
│   │   └── standard_scaler.joblib          # Padronizador ajustado exclusivamente nos dados de treino
│   ├── figures/                            # Gráficos exportados em alta resolução (200 DPI)
│   │   ├── distribuicao_notas_qualidade.png
│   │   ├── distribuicoes_histogramas.png
│   │   ├── distribuicoes_boxplots.png
│   │   ├── matriz_correlacao.png
│   │   ├── comparacao_por_classe.png
│   │   └── feature_so2_ratio.png
│   └── tables/
│       └── resumo_outliers_iqr.csv         # Relatório quantitativo de outliers por variável
│
├── requirements.txt                        # Dependências e bibliotecas utilizadas
└── README.md                               # Documentação executiva e técnica do projeto
```

## 🚀 6. Como Executar o Projeto Localmente

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/and-o-re/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. **Crie e ative um ambiente virtual:**

   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute os Notebooks na ordem da pipeline:**
   Abra o projeto no VS Code ou Jupyter Lab e execute sequencialmente:

   1. `notebooks/01_eda.ipynb`
   2. `notebooks/02_preprocessamento.ipynb`
   3. `notebooks/03_modelo.ipynb` *(ou `notebooks/03_modelo_v2.ipynb`)*

## 👥 7. Autores (Equipe)

Projeto desenvolvido para a Fase 2 da Pós-Graduação em Data Analytics (**POS TECH — FIAP**):

* **André Carlos** — [GitHub](https://github.com/and-o-re)
* **Anne Pimentel** — [GitHub](https://github.com/anneaiad)
* **Ewerton Reis** — [GitHub](https://github.com/ewpreis-noetrix)
