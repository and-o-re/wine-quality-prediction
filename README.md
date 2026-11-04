# 🍷 Classificando a Qualidade de Vinhos com Machine Learning
**POS TECH — Data Analytics (DTAT) | Tech Challenge - Fase 2**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

---

## 📌 1. Contexto e Compreensão do Problema

A indústria vitivinícola possui grande relevância econômica e cultural em diversos países. Tradicionalmente, a avaliação da qualidade de um vinho é realizada por especialistas por meio de análises sensoriais (avaliando aroma, sabor, acidez e equilíbrio). Contudo, esse processo pode ser subjetivo, demandar bastante tempo e depender fortemente da experiência dos avaliadores.

Com o avanço das técnicas de Ciência de Dados e Aprendizado de Máquina, tornou-se possível utilizar dados físico-químicos obtidos durante a produção para auxiliar na previsão da qualidade final da bebida. Modelos preditivos apoiam enólogos e produtores na tomada de decisão, permitindo ajustes rápidos no processo produtivo e contribuindo para a padronização da qualidade.

### 🎯 Objetivo do Projeto
Desenvolver uma pipeline completa de análise exploratória, pré-processamento e modelagem preditiva capaz de classificar a qualidade de um vinho com base em suas propriedades físico-químicas.

Para fins de modelagem, a nota de qualidade original (`quality`, atribuída de 0 a 10) foi transformada em uma **variável alvo binária**:
* **Vinho de Alta Qualidade (`1`):** $\text{nota} \ge 7$
* **Vinho de Baixa/Média Qualidade (`0`):** $\text{nota} < 7$

---

## 📊 2. Sobre o Conjunto de Dados (Dataset)

O projeto utiliza a base pública **Wine Quality Dataset** (disponível no [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)), que contempla as seguintes variáveis físico-químicas e sensoriais:

| Variável | Descrição | Tipo |
| :--- | :--- | :--- |
| `fixed acidity` | Acidez fixa (ácidos não voláteis presentes no vinho, como o tartárico) | Numérica Contínua |
| `volatile acidity` | Acidez volátil (quantidade de ácido acético; níveis altos geram sabor avinagrado) | Numérica Contínua |
| `citric acid` | Ácido cítrico (adiciona frescor e sabor aos vinhos) | Numérica Contínua |
| `residual sugar` | Açúcar residual (quantidade de açúcar restante após o término da fermentação) | Numérica Contínua |
| `chlorides` | Cloretos (quantidade de sal presente no vinho) | Numérica Contínua |
| `free sulfur dioxide` | Dióxido de enxofre livre ($SO_2$ livre; previne oxidação e crescimento microbiano) | Numérica Contínua |
| `total sulfur dioxide` | Dióxido de enxofre total (soma das formas livre e ligada de $SO_2$) | Numérica Contínua |
| `density` | Densidade do vinho (varia conforme o percentual de álcool e teor de açúcar) | Numérica Contínua |
| `pH` | Escala de acidez ou basicidade do vinho (geralmente entre $3.0$ e $4.0$) | Numérica Contínua |
| `sulphates` | Sulfatos (aditivo que contribui para os níveis de $SO_2$, atuando como antimicrobiano) | Numérica Contínua |
| `alcohol` | Teor alcoólico percentual por volume | Numérica Contínua |
| `quality` | **Variável Alvo:** Nota sensorial atribuída por especialistas (convertida em classe binária) | Categórica / Alvo |

---

## 🛠️ 3. Pipeline do Desafio

O desenvolvimento deste trabalho foi dividido em 6 etapas principais:

1. **Compreensão do Problema:** Definição da regra de negócio e binarização da variável alvo ($\text{quality} \ge 7$).
2. **Análise Exploratória de Dados (EDA):**
   * Estudo da distribuição estatística das variáveis físico-químicas.
   * Análise da matriz de correlação e justificativa química/enológica das relações observadas.
   * Detecção e investigação de *outliers* e valores inconsistentes.
   * Avaliação do desbalanceamento entre as classes de vinhos de alta e baixa/média qualidade.
3. **Pré-processamento de Dados:**
   * Limpeza e tratamento de registros duplicados ou ausentes.
   * Padronização/Normalização das variáveis numéricas (`StandardScaler` / `MinMaxScaler`).
   * Engenharia de atributos (*Feature Engineering*) e tratamento de desbalanceamento (ex.: `SMOTE` ou ajuste de pesos de classe, quando aplicável).
4. **Desenvolvimento de Modelos:**
   * Treinamento e validação cruzada de múltiplos algoritmos de classificação (ex.: *Logistic Regression*, *Random Forest*, *Gradient Boosting / XGBoost*, etc.).
5. **Avaliação dos Modelos:**
   * Comparação de performance utilizando métricas adequadas para dados desbalanceados: **Acurácia, Precisão (Precision), Revocação (Recall), F1-Score e ROC-AUC**.
6. **Interpretação dos Resultados:**
   * Identificação da importância das variáveis (*Feature Importance*) na definição de um vinho premium.
   * Recomendações práticas para o processo produtivo na vinícola.

---

## 📁 4. Estrutura do Repositório

O repositório está organizado seguindo as boas práticas recomendadas pelo desafio:

```text
wine-quality-prediction/
│
├── data/               # Base de dados utilizada (bruta e processada)
├── notebooks/          # Jupyter Notebooks com a Análise Exploratória (EDA) e Modelagem
├── src/                # Scripts auxiliares (funções de pré-processamento, treino e avaliação)
├── results/            # Gráficos gerados, matrizes de confusão, métricas e apresentação executiva
├── requirements.txt    # Lista de bibliotecas e dependências do projeto
└── README.md           # Documentação principal do projeto
```

---

## 🚀 5. Como Executar o Projeto Localmente

### Pré-requisitos
* **Python 3.10+** instalado
* **Git** instalado

### Passo a passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/and-o-re/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute os Notebooks:**
   Inicie o Jupyter Lab ou Jupyter Notebook para explorar as análises:
   ```bash
   jupyter notebook
   ```
   Acesse a pasta `notebooks/` e execute as células sequencialmente.

---

## 📈 6. Principais Resultados e Insights

> *(Preencha ou ajuste os números abaixo com as métricas finais obtidas no seu notebook)*

### Comparativo de Modelos
| Modelo | Acurácia | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modelo 1 (ex: Regressão Logística)** | `0.XX` | `0.XX` | `0.XX` | `0.XX` | `0.XX` |
| **Modelo 2 (ex: Random Forest)** | `0.XX` | `0.XX` | `0.XX` | `0.XX` | `0.XX` |
| **Modelo 3 (ex: XGBoost)** | `0.XX` | `0.XX` | `0.XX` | `0.XX` | `0.XX` |

### Variáveis de Maior Influência na Qualidade
* **Teor Alcoólico (`alcohol`):** Apresenta forte correlação positiva com vinhos de alta qualidade ($\text{nota} \ge 7$), refletindo maior maturação das uvas e fermentação eficiente.
* **Acidez Volátil (`volatile acidity`):** Forte correlação negativa; níveis elevados de ácido acético degradam o aroma e conferem gosto avinagrado, reduzindo drasticamente a chance de o vinho ser classificado como alta qualidade.
* **Sulfatos (`sulphates`) e Ácido Cítrico (`citric acid`):** Contribuem positivamente para a conservação, frescor e estrutura sensorial do vinho.

---

## 🎬 7. Entregáveis Executivos

* 📄 **Apresentação Executiva (Storytelling da EDA e Resultados):** [Acesse aqui o arquivo PDF/PPT na pasta `results/`](./results/) *(atualizar com o nome exato do arquivo)*
* 🎥 **Vídeo Executivo (Pitch de até 5 min):** [Assista à apresentação em vídeo aqui](#) *(inserir link do YouTube/Drive)*

---

## 👥 8. Autores (Equipe)

Projeto desenvolvido para a Fase 2 da Pós-Graduação em Data Analytics (**POS TECH - FIAP**):

* **[Andre Carlos]** — [GitHub](https://github.com/and-o-re)
* **[Anne Pimentel]** — [GitHub](https://github.com/anneaiad)
* **[Ewerton Reis]** — [GitHub](https://github.com/ewpreis-noetrix)
