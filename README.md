Aqui está o README atualizado com a **Semana 2 — Pré-processamento e baseline de Machine Learning** incluída:

````md
# 📰 ML News Classification

> Projeto em desenvolvimento para construção de um classificador de notícias usando Machine Learning, com foco em aprendizado prático de aplicações de ML e MLOps.

## 📌 Sobre o projeto

O **ML News Classification** é um projeto end-to-end de classificação automática de notícias.

O objetivo é construir uma pipeline completa de Machine Learning, passando por:

- Coleta e preparação dos dados
- Análise exploratória
- Pré-processamento textual
- Treinamento de modelos
- Avaliação de desempenho
- Comparação entre modelos
- Versionamento de experimentos
- Organização do código para produção
- Práticas de MLOps

Este projeto ainda está em desenvolvimento e será evoluído de forma incremental.

---

## 🎯 Objetivos

- Aprender aplicações práticas de Machine Learning em NLP
- Criar um classificador de notícias por categoria
- Estruturar uma pipeline de ML reutilizável
- Aplicar boas práticas de MLOps
- Construir um projeto forte para portfólio

---

## 🧠 Problema

Dado o texto de uma notícia, o sistema deverá prever automaticamente sua categoria.

O texto utilizado para treinamento é formado pela combinação das colunas:

```python
headline + " " + short_description
````

A categoria da notícia é usada como variável alvo do modelo.

Exemplos de categorias presentes no dataset:

* Politics
* Wellness
* Entertainment
* Travel
* Style & Beauty
* Parenting
* Food & Drink
* Business
* Sports
* Technology

---

## 🗂️ Dataset

O projeto utiliza um dataset de notícias contendo informações como:

* Link da notícia
* Título
* Categoria
* Descrição curta
* Autor
* Data de publicação

Exemplo de registro:

```json
{
  "link": "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9",
  "headline": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
  "category": "U.S. NEWS",
  "short_description": "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.",
  "authors": "Carla K. Johnson, AP",
  "date": "2022-09-23"
}
```

---

## 🛠️ Stack utilizada

### Linguagem

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### NLP

* TF-IDF
* Bag of Words
* Normalização textual

### Visualização e análise

* Matplotlib
* Seaborn
* Jupyter Notebook

### Persistência de modelos

* Joblib

### MLOps

* MLflow para tracking de experimentos
* DVC para versionamento de dados
* Docker para containerização
* GitHub Actions para CI/CD

### API e deploy

* FastAPI
* Uvicorn
* Docker
* Deploy futuro em Render, Railway ou similar

### Qualidade de código

* Pytest
* Ruff
* Black
* Pre-commit

---

## 📁 Estrutura do projeto

```bash
ml-news-classification/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
├── models/
│   └── news_classifier.joblib
├── notebooks/
│   └── 01_eda_news_dataset.ipynb
├── reports/
│   └── metrics/
│       ├── classification_report.txt
│       ├── confusion_matrix.csv
│       └── confusion_matrix.png
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── eda_news_dataset.py
│   │   └── load_data.py
│   └── models/
│       ├── train_model.py
│       └── evaluate_model.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚧 Status do projeto

Projeto em andamento.

Atualmente, o projeto possui:

* Dataset carregado e analisado
* Dados tratados e separados em treino e teste
* Pipeline TF-IDF criado
* Modelo baseline com Naive Bayes treinado
* Modelos clássicos treinados e comparados
* Avaliação com métricas de classificação
* Matriz de confusão gerada
* Scripts iniciais organizados em `src/`
* Modelo treinado salvo em `models/`

---

## 📊 Métricas utilizadas

As principais métricas usadas no projeto são:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Essas métricas ajudam a avaliar não apenas a taxa geral de acerto, mas também o desempenho do modelo em cada categoria.

---

## 🚀 Como rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/ml-news-classification.git
```

### 2. Acesse a pasta do projeto

```bash
cd ml-news-classification
```

### 3. Crie o ambiente virtual

```bash
python -m venv .venv
```

### 4. Ative o ambiente virtual

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/Mac:

```bash
source .venv/bin/activate
```

### 5. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🧪 Como executar os scripts

### Gerar relatório de EDA

```bash
python -m src.data.eda_news_dataset
```

### Treinar o modelo

```bash
python -m src.models.train_model
```

Esse comando treina o pipeline de classificação e salva o modelo em:

```text
models/news_classifier.joblib
```

### Avaliar o modelo

```bash
python -m src.models.evaluate_model
```

Esse comando gera os arquivos de avaliação em:

```text
reports/metrics/
```

Incluindo:

```text
classification_report.txt
confusion_matrix.csv
confusion_matrix.png
```

---

## 🤖 Modelos implementados

Até o momento, foram utilizados modelos clássicos de Machine Learning:

* Naive Bayes
* Logistic Regression
* Linear SVM

O pipeline utiliza `TfidfVectorizer` para transformar os textos em representações numéricas antes do treinamento.

---

## 📌 Roadmap

* [x] Versão 0.1: Setup inicial e escolha do dataset
* [x] Versão 0.2: Pipeline de pré-processamento
* [x] Versão 0.3: Modelo baseline
* [x] Versão 0.4: Comparação com modelos clássicos
* [x] Versão 0.5: Organização inicial dos scripts em `src/`
* [ ] Versão 0.6: Tracking de experimentos com MLflow
* [ ] Versão 0.7: Versionamento de dados com DVC
* [ ] Versão 0.8: Testes automatizados
* [ ] Versão 0.9: API com FastAPI
* [ ] Versão 1.0: Pipeline completa com Docker e deploy

---

## 🌿 GitHub Flow

Este projeto segue o modelo **GitHub Flow**, um fluxo de trabalho simples e eficiente para desenvolvimento contínuo.

### 🔄 Como funciona

1. Criação de uma nova branch a partir da `main`
2. Desenvolvimento da feature ou correção
3. Commits pequenos e descritivos
4. Abertura de Pull Request
5. Revisão e validação
6. Merge na `main`

### 📌 Convenções adotadas

Branches nomeadas por tipo:

* `feat/...` → novas funcionalidades
* `fix/...` → correções de bugs
* `chore/...` → tarefas gerais
* `docs/...` → documentação
* `refactor/...` → refatorações de código

Commits seguem padrão semântico, usando Conventional Commits:

* `feat: ...`
* `fix: ...`
* `docs: ...`
* `refactor: ...`
* `chore: ...`

---

## 📄 Licença

Este projeto está em desenvolvimento para fins de estudo e portfólio.

````

Commit sugerido:

```bash
git add README.md
git commit -m "docs: update readme with week 2 progress"
````
