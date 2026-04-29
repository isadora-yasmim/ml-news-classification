# 📰 ML News Classification

> Projeto em desenvolvimento para construção de um classificador de notícias usando Machine Learning, com foco em aprendizado prático de aplicações de ML e MLOps.

## 📌 Sobre o projeto

O **ML News Classification** é um projeto end-to-end de classificação automática de notícias.

O objetivo é construir uma pipeline completa de Machine Learning, passando por:

- Coleta e preparação dos dados
- Análise exploratória
- Treinamento de modelos
- Avaliação de desempenho
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

Exemplos de categorias:

- Política
- Economia
- Esportes
- Tecnologia
- Saúde
- Entretenimento
- Mundo

---

## 🛠️ Stack prevista

### Linguagem

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy

### NLP

- TF-IDF
- Bag of Words
- Embeddings futuramente

### Visualização e análise

- Matplotlib
- Seaborn
- Jupyter Notebook

### MLOps

- MLflow para tracking de experimentos
- DVC para versionamento de dados
- Docker para containerização
- GitHub Actions para CI/CD

### API e deploy

- FastAPI
- Uvicorn
- Docker
- Deploy futuro em Render, Railway ou similar

### Qualidade de código

- Pytest
- Ruff
- Black
- Pre-commit

---

## 📁 Estrutura inicial prevista

```bash
ml-news-classification/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── api/
├── tests/
├── models/
├── reports/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
````

---

## 🚧 Status do projeto

Projeto em andamento.

Atualmente, o foco está na definição da arquitetura, stack e backlog inicial.

---

## ✅ Backlog

### 1. Setup inicial do projeto

* [ ] Criar estrutura de pastas
* [ ] Configurar ambiente virtual
* [ ] Criar `.gitignore`
* [ ] Criar `README.md`
* [ ] Definir dependências iniciais
* [ ] Configurar lint e formatter

### 2. Dados

* [ ] Escolher dataset de notícias
* [ ] Armazenar dados brutos em `data/raw`
* [ ] Criar script de carregamento dos dados
* [ ] Criar etapa de limpeza textual
* [ ] Criar versão processada dos dados

### 3. Análise exploratória

* [ ] Analisar distribuição das categorias
* [ ] Verificar tamanho médio dos textos
* [ ] Identificar dados faltantes
* [ ] Visualizar frequência de palavras
* [ ] Documentar principais insights

### 4. Modelagem baseline

* [ ] Criar baseline com TF-IDF
* [ ] Treinar Logistic Regression
* [ ] Treinar Naive Bayes
* [ ] Treinar Random Forest
* [ ] Comparar resultados iniciais

### 5. Avaliação

* [ ] Usar métricas como accuracy, precision, recall e F1-score
* [ ] Criar matriz de confusão
* [ ] Avaliar erros por categoria
* [ ] Salvar relatório de avaliação

### 6. MLOps

* [ ] Configurar MLflow
* [ ] Registrar experimentos
* [ ] Versionar datasets com DVC
* [ ] Salvar modelos treinados
* [ ] Criar pipeline reprodutível de treino
* [ ] Configurar testes automatizados
* [ ] Configurar GitHub Actions

### 7. API

* [ ] Criar API com FastAPI
* [ ] Criar endpoint de predição
* [ ] Carregar modelo treinado na API
* [ ] Criar validação com Pydantic
* [ ] Testar endpoint localmente

### 8. Docker e deploy

* [ ] Criar Dockerfile
* [ ] Criar docker-compose
* [ ] Rodar API via container
* [ ] Fazer deploy da aplicação
* [ ] Documentar como executar o projeto

### 9. Melhorias futuras

* [ ] Testar modelos com embeddings
* [ ] Usar modelos Transformer
* [ ] Criar interface simples para classificação
* [ ] Adicionar monitoramento de dados
* [ ] Adicionar monitoramento de performance do modelo

---

## 📊 Métricas previstas

As principais métricas usadas serão:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🚀 Como rodar o projeto

> Esta seção será atualizada conforme o projeto for implementado.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ml-news-classification.git

# Acesse a pasta
cd ml-news-classification

# Crie o ambiente virtual
python -m venv .venv

# Ative o ambiente virtual
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

---

## 📌 Roadmap

* [ ] Versão 0.1: Setup inicial e escolha do dataset
* [ ] Versão 0.2: Pipeline de pré-processamento
* [ ] Versão 0.3: Modelo baseline
* [ ] Versão 0.4: Tracking com MLflow
* [ ] Versão 0.5: API com FastAPI
* [ ] Versão 1.0: Pipeline completa com Docker e deploy
---

## 🌿 GitHub Flow

Este projeto segue o modelo **GitHub Flow**, um fluxo de trabalho simples e eficiente para desenvolvimento contínuo.

### 🔄 Como funciona

1. Criação de uma nova branch a partir da `main`
2. Desenvolvimento da feature ou correção
3. Commits pequenos e descritivos
4. Abertura de Pull Request (PR)
5. Revisão e validação
6. Merge na `main`

### 📌 Convenções adotadas

- Branches nomeadas por tipo:
  - `feat/...` → novas funcionalidades
  - `fix/...` → correções de bugs
  - `chore/...` → tarefas gerais
  - `docs/...` → documentação

- Commits seguem padrão semântico (Conventional Commits):
  - `feat: ...`
  - `fix: ...`
  - `docs: ...`
  - `refactor: ...`

