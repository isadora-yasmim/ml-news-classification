# 🗓️ Cronograma de Execução — ML News Classification

## 🎯 Objetivo do projeto

Construir um classificador de notícias com Machine Learning, utilizando uma base de dados rotulada, aplicando boas práticas de desenvolvimento, avaliação de modelos e conceitos iniciais de MLOps.

---

# ✅ Semana 1 — Planejamento, dados e estrutura do projeto

## Objetivo da semana
Preparar o repositório, entender o dataset e deixar a base pronta para experimentação.

## Checklist

### Organização do repositório
- [x] Criar o repositório `ml-news-classification`
- [x] Criar estrutura inicial de pastas
- [x] Adicionar `README.md`
- [x] Adicionar `.gitignore`
- [x] Definir stack inicial do projeto
- [x] Documentar que o projeto usa GitHub Flow

### Estrutura sugerida
- [x] Criar pasta `data/`
- [x] Criar pasta `notebooks/`
- [x] Criar pasta `src/`
- [x] Criar pasta `models/`
- [x] Criar pasta `reports/`
- [x] Criar pasta `tests/`

### [Dataset](relatorio_dataset.md)
- [x] Baixar dataset de notícias
- [x] Conferir colunas disponíveis
- [x] Identificar coluna de texto
- [x] Identificar coluna de categoria
- [x] Verificar quantidade de categorias
- [x] Verificar quantidade de registros por categoria
- [x] Verificar valores nulos
- [x] Verificar textos duplicados

### [Análise exploratória](notebooks\01_eda_news_dataset.ipynb)
- [x] Analisar distribuição das categorias
- [x] Analisar tamanho médio dos textos
- [x] Verificar categorias muito desbalanceadas
- [x] Decidir se vai usar apenas `headline` ou `headline + short_description`
- [x] Criar notebook de EDA

### Entregáveis da semana
- [x] Repositório organizado
- [x] Dataset armazenado localmente
- [x] Notebook inicial de análise exploratória
- [x] README inicial atualizado
- [x] Primeiro commit da estrutura base

---

# ✅ Semana 2 — Pré-processamento e baseline de Machine Learning

## Objetivo da semana
Criar o primeiro pipeline funcional de classificação usando modelos clássicos de ML.

## Checklist

### Preparação dos dados
- [x] Carregar dataset com Pandas
- [x] Remover registros com texto vazio
- [x] Remover registros sem categoria
- [x] Criar coluna `text` juntando `headline + short_description`
- [x] Criar coluna `label` com a categoria
- [x] Normalizar textos
- [x] Separar dados em treino e teste
- [x] Garantir split estratificado por categoria

### Baseline
- [x] Criar pipeline com `TfidfVectorizer`
- [x] Treinar modelo Naive Bayes
- [x] Avaliar baseline
- [x] Salvar métricas iniciais

### Modelos clássicos
- [x] Treinar Logistic Regression
- [x] Treinar Linear SVM
- [x] Comparar modelos
- [x] Gerar `classification_report`
- [x] Gerar matriz de confusão
- [x] Identificar categorias com pior desempenho

### Organização do código
- [x] Transformar notebook em scripts
- [x] Criar script de treinamento
- [x] Criar script de avaliação
- [x] Criar arquivo de configuração simples
- [x] Salvar modelo treinado em `models/`

### Entregáveis da semana
- [x] Primeiro modelo funcional
- [x] Métricas de baseline documentadas
- [x] Comparação entre modelos
- [x] Scripts iniciais em `src/`
- [x] README atualizado com como treinar o modelo

---

# ✅ Semana 3 — Melhorias, experimentos e versionamento

## Objetivo da semana
Melhorar a qualidade do modelo, organizar experimentos e iniciar práticas de MLOps.

## Checklist

### Melhorias no modelo
- [x] Testar diferentes parâmetros do TF-IDF
- [x] Testar `ngram_range`
- [x] Testar limite de features
- [x] Testar remoção de stopwords
- [x] Testar balanceamento de classes
- [x] Rodar GridSearchCV 

### Avaliação
- [x] Comparar métricas entre versões
- [x] Usar Accuracy
- [x] Usar Precision
- [x] Usar Recall
- [x] Usar F1-score
- [x] Priorizar F1-score macro se houver desbalanceamento
- [x] Gerar gráficos de comparação

### Versionamento e rastreabilidade
- [x] Versionar experimentos com arquivos `.json` ou `.csv`
- [x] Salvar hiperparâmetros usados
- [x] Salvar métricas por execução
- [x] Definir melhor modelo até o momento
- [x] Criar pasta `reports/figures/`

### MLOps inicial
- [x] Adicionar `requirements.txt` ou `pyproject.toml`
- [x] Criar script reproduzível de treino
- [x] Criar script reproduzível de avaliação
- [x] Adicionar instruções de instalação no README
- [x] Adicionar instruções para reprodução dos resultados

### Entregáveis da semana
- [x] Modelo otimizado
- [x] Registro de experimentos
- [x] Gráficos de avaliação
- [x] Melhor modelo salvo
- [x] README com resultados parciais

---

# ✅ Semana 4 — API, documentação final e entrega

## Objetivo da semana
Transformar o modelo em uma aplicação utilizável e finalizar a documentação do projeto.

## Checklist

### Interface de uso
- [x] Criar função de predição
- [x] Permitir classificar uma notícia nova
- [x] Criar entrada com `headline`
- [x] Criar entrada com `short_description`
- [x] Retornar categoria prevista
- [x] Retornar probabilidade/confiança, se o modelo permitir


#### API com FastAPI
- [x] Criar endpoint `/predict`
- [x] Criar endpoint `/health`
- [x] Criar schema de entrada
- [x] Criar schema de saída
- [x] Testar API localmente


### Testes
- [x] Criar testes básicos para pré-processamento
- [x] Criar teste para carregamento do modelo
- [x] Criar teste para predição
- [x] Testar comportamento com texto vazio
- [x] Testar comportamento com texto muito curto

### Documentação final
- [ ] Atualizar README completo
- [ ] Adicionar descrição do problema
- [ ] Adicionar descrição do dataset
- [ ] Adicionar stack utilizada
- [ ] Adicionar estrutura de pastas
- [ ] Adicionar como instalar
- [ ] Adicionar como treinar
- [ ] Adicionar como avaliar
- [ ] Adicionar como executar predição
- [ ] Adicionar tabela de resultados
- [ ] Adicionar próximos passos

### Finalização
- [ ] Revisar código
- [ ] Remover arquivos desnecessários
- [ ] Conferir commits
- [ ] Criar PR final
- [ ] Escrever descrição da PR
- [ ] Fazer merge na `main`

### Entregáveis da semana
- [ ] Projeto funcional
- [ ] Modelo treinado salvo
- [ ] API ou CLI funcionando
- [ ] Testes básicos criados
- [ ] README finalizado
- [ ] Projeto pronto para portfólio

---

# 📌 Resultado esperado ao final das 4 semanas

Ao final do cronograma, o projeto deve conter:

- [x] Dataset tratado
- [x] Análise exploratória
- [x] Pipeline de treinamento
- [x] Modelo baseline
- [ ] Modelo otimizado
- [ ] Avaliação com métricas
- [ ] Gráficos de resultados
- [ ] Modelo salvo
- [ ] API ou CLI para predição
- [ ] Documentação completa
- [ ] Fluxo organizado com GitHub Flow

---

# 🚀 Extras para deixar o projeto mais forte

Se sobrar tempo:

- [x] Adicionar Docker
- [ ] Adicionar MLflow
- [ ] Adicionar DVC para versionamento de dados
- [ ] Criar GitHub Actions para testes
- [ ] Criar deploy simples da API
- [ ] Testar modelo com BERT
- [ ] Criar dashboard simples com Streamlit
- [ ] Adicionar monitoramento de drift