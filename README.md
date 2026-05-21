# 📰 ML News Classification

<p align="center">
  Classificação automática de notícias utilizando Machine Learning, NLP e FastAPI.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" />
  <img src="https://img.shields.io/badge/fastapi-API-green" />
  <img src="https://img.shields.io/badge/status-em%20desenvolvimento-yellow" />
  <img src="https://img.shields.io/badge/license-educational-lightgrey" />
</p>

---

# 📌 Visão Geral

O **ML News Classification** é um projeto de classificação automática de notícias utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Machine Learning supervisionado**.

O sistema recebe uma notícia composta por:

* `headline`
* `short_description`

E retorna automaticamente a categoria prevista da notícia.

Além da modelagem preditiva, o projeto também possui foco em:

* organização de pipelines de ML
* versionamento de experimentos
* testes automatizados
* APIs para inferência
* boas práticas de engenharia de software
* fundamentos de MLOps

---

# 🚀 Funcionalidades

* Classificação automática de notícias
* Pipeline de NLP com TF-IDF
* Modelos clássicos de Machine Learning
* API REST com FastAPI
* Sistema de inferência desacoplado
* Testes automatizados
* Versionamento de experimentos
* Estrutura escalável para evolução futura

---

# 📚 Índice

* [Problema](#-problema)
* [Dataset](#-dataset)
* [Arquitetura](#-arquitetura)
* [Stack Utilizada](#-stack-utilizada)
* [Estrutura do Projeto](#-estrutura-do-projeto)
* [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
* [Instalação](#-instalação)
* [Treinamento](#-treinamento)
* [Avaliação](#-avaliação)
* [Execução da API](#-execução-da-api)
* [Predição](#-predição)
* [Testes](#-testes)
* [Resultados](#-resultados)
* [Experimentos](#-experimentos)
* [Próximos Passos](#-próximos-passos)
* [Licença](#-licença)

---

# 🎯 Problema

Portais de notícias produzem milhares de conteúdos diariamente. Realizar a categorização manual desses conteúdos é um processo:

* custoso
* demorado
* pouco escalável

Este projeto busca automatizar esse processo utilizando aprendizado de máquina supervisionado.

---

# 🗂 Dataset

O projeto utiliza o dataset de notícias do HuffPost.

| Campo               | Descrição            |
| ------------------- | -------------------- |
| `headline`          | Título da notícia    |
| `short_description` | Resumo da notícia    |
| `category`          | Categoria da notícia |
| `authors`           | Autor(es)            |
| `date`              | Data de publicação   |
| `link`              | Link da notícia      |

## Estratégia textual

```python
headline + " " + short_description
```

## Volume de dados

* +200 mil notícias
* 42 categorias
* Dados textuais reais

---

# 🏗 Arquitetura

```text
Texto → Pré-processamento → TF-IDF → Modelo ML → API → Predição
```

---

# 🛠 Stack Utilizada

## Linguagem

* Python 3.14

## Bibliotecas principais

### Data Science

* Pandas
* NumPy
* Scikit-learn

### API

* FastAPI
* Uvicorn

### Testes

* Pytest
* HTTPX

### Persistência

* Joblib

## Ferramentas

* Git
* GitHub Flow
* VS Code

---

# 📁 Estrutura do Projeto

```text
ml-news-classification/
│
├── data/
├── experiments/
├── models/
├── notebooks/
├── src/
│   ├── api/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── utils/
├── tests/
├── requirements.txt
└── README.md
```

---

# 🤖 Pipeline de Machine Learning

## Pré-processamento

* remoção de valores nulos
* remoção de textos vazios
* normalização textual
* split estratificado

## Vetorização

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)
```

## Modelos treinados

* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM

## Avaliação

* Accuracy
* Classification Report
* Confusion Matrix

---

# ⚙ Instalação

## Clonar repositório

```bash
git clone https://github.com/seu-usuario/ml-news-classification.git
```

## Criar ambiente virtual

```bash
python -m venv .venv
```

## Ativar ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## Instalar dependências

```bash
pip install -r requirements.txt
```

---

# 🧠 Treinamento

## Naive Bayes

```bash
python src/models/train_naive_bayes.py
```

## Logistic Regression

```bash
python src/models/train_logistic_regression.py
```

## Linear SVM

```bash
python src/models/train_linear_svm.py
```

---

# 📊 Avaliação

Os relatórios de métricas e experimentos são armazenados em:

```text
experiments/
```

---

# 🌐 Execução da API

```bash
uvicorn src.api.main:app --reload
```

## Swagger/OpenAPI

```text
http://127.0.0.1:8000/docs
```

---

# 🔮 Predição

## Endpoint `/predict`

### Entrada

```json
{
  "headline": "New technology is changing the way people work",
  "short_description": "Companies are adopting artificial intelligence tools to improve productivity."
}
```

### Saída

```json
{
  "predicted_category": "TECH",
  "confidence": 0.7321
}
```

---

# 🧪 Testes

```bash
pytest -v
```

Testes implementados:

* carregamento do modelo
* predição
* API FastAPI
* validação de entrada
* texto vazio
* texto curto

---

# 📈 Resultados

| Modelo              | Accuracy     |
| ------------------- | ------------ |
| Naive Bayes         | 0.5221       |
| Logistic Regression | Em avaliação |
| Linear SVM          | Em avaliação |

---

# 🧾 Experimentos

Os experimentos são versionados em arquivos `.json` contendo:

* modelo
* hiperparâmetros
* métricas
* timestamp

---

# 🔭 Próximos Passos

* Docker
* CI/CD
* Deploy em nuvem
* Monitoramento de modelos
* Transformers
* Frontend para inferência
* Autenticação na API

---

# ✅ Boas práticas utilizadas

* GitHub Flow
* Conventional Commits
* Arquitetura modular
* Testes automatizados
* Versionamento de experimentos

---

# 📄 Licença

Projeto desenvolvido para fins educacionais e estudo de Machine Learning, NLP e MLOps.
