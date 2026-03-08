# Bank Product Usage Profiling with AI
#Clusterização + Propensão + Dinâmica Temporal (Markov)

Demo (Streamlit Cloud): https://bank-customer-profiling-ai-rj8rbhntnd2r6ksq5cb37s.streamlit.app

---

## Executive Summary

Solução analítica end-to-end para perfilamento comportamental de clientes no setor bancário, combinando:

- Aprendizado não supervisionado (Clusterização) para segmentação comportamental  
- Aprendizado supervisionado (Modelagem de Propensão) para previsão de adoção de produtos  
- Cadeias de Markov para modelagem das transições de estado dos clientes ao longo do tempo  
- Modelo de propensão por produto (supervisionado)
- Survival Analysis (tempo até adoção)
 
A solução permite tomada de decisão orientada por dados para:

- Estratégia de produtos  
- Direcionamento de campanhas de Cross-sell e Up-sell  
- Monitoramento de risco  
- Análise do ciclo de vida do cliente  

---

## 🎯 Objetivo

Demonstrar como técnicas de Machine Learning podem ser utilizadas para:

- Identificar perfis latentes de clientes
- Estimar probabilidade de adoção de produtos financeiros
- Modelar o tempo até contratação
- Modelar a dinâmica temporal de migração entre estados comportamentais
 
Os dados utilizados são sintéticos, porém estruturados de forma realista.

---

## 🏗️ Pipeline

1. **Geração de dados sintéticos**
   - Painel mensal por cliente (12 meses)
   - Variáveis financeiras e comportamentais
   - Evento de adoção de investimento
   - Variáveis para Survival Analysis

2. **Feature Engineering**
   - Agregação por cliente
   - Construção de variáveis comportamentais
   - Taxas de atraso e uso de crédito

3. **Clusterização (K-Means)**
   - Padronização das variáveis
   - Identificação de perfis latentes
   - Avaliação via Silhouette Score

5. **Modelo de Propensão**
   - Target: adoção de produto de investimento em até 3 meses
   - Features: comportamento inicial (janela 1–3 meses)
   - Inclusão do cluster como variável latente
📌 Métricas avaliadas:
   - AUC
   - Recall@10%
   - Recall@20%
Saída:
   - propensity_metrics.csv
   - propensity_scores.csv
  
6. **Dinâmica Temporal (Cadeias de Markov)**
   - Definição de estados comportamentais mensais (Low / Medium / High)
   - Estimativa da matriz de transição:

𝑃(𝑆𝑡𝑎𝑡𝑒𝑡+1∣𝑆𝑡𝑎𝑡𝑒𝑡)

   - Cálculo da distribuição estacionária (steady state)

📌 Saídas:
   - markov_transition_matrix.csv
   - markov_steady_state.csv

  
7. **Survival Analysis**
 
---

## 👥 Perfis Identificados

Resumo disponível em:

reports/cluster_profile_cards.md
Os clusters representam perfis distintos baseados em:
- Renda
- Intensidade de uso digital
- Exposição a crédito
- Risco de inadimplência

Exemplo de perfis identificados:

- Digital Crédito Intensivo
- Alta Renda Estável
- Conservador Tradicional
- Digital Estável

---

## 📊 Métrica de Clusterização

- k = 4
- Silhouette ≈ 0.22

Valor consistente com segmentações reais em dados financeiros.

---
📊 Principais Insights
 - Convergência estrutural para estado intermediário (Medium)
 - Estados extremos apresentam baixa estabilidade temporal
 - Modelo de propensão requer engenharia adicional de variáveis para ganho de separabilidade
 - Integração entre cluster + propensão + Markov permite visão completa do ciclo de vida do cliente

---

## 🧠 Técnicas Utilizadas

- Python
- Pandas / NumPy
- Scikit-learn
- K-Means
- StandardScaler
- Modelagem probabilística
- Survival Analysis (Cox Model)

---

## 🚀 Como Executar

```bash
pip install -r requirements.txt
python -m src.generate_data
python -m src.build_features
python -m src.cluster_profiles
```

📁 Estrutura do Projeto

src/
data/
reports/
docs/

## Projeto

bank-customer-profiling-ai
│
├── data
│   ├── raw
│   └── processed
│
├── src
├── models
├── reports
└── README.md

## Business Value

Este projeto demonstra como técnicas de **Data Science e Machine Learning** podem apoiar decisões estratégicas no setor bancário, especialmente na análise de comportamento e ciclo de vida de clientes.

A solução integra diferentes abordagens analíticas para responder perguntas relevantes para instituições financeiras:

### 1. Segmentação de Clientes
Através de **clusterização (K-Means)**, os clientes são agrupados em perfis comportamentais com base em:

- renda
- saldo
- uso de cartão
- comportamento digital (PIX)
- risco de atraso

Isso permite identificar segmentos como:
- clientes de alta renda
- usuários intensivos de crédito
- perfis digitais
- clientes conservadores

### 2. Dinâmica de Comportamento (Markov Chains)
As **cadeias de Markov** permitem modelar como clientes transitam entre perfis ao longo do tempo.

Isso ajuda a responder perguntas como:
- clientes estão migrando para perfis mais digitais?
- quais segmentos são mais estáveis?
- quais perfis apresentam maior mudança comportamental?

### 3. Propensão à Adoção de Investimentos
Um modelo supervisionado estima a **probabilidade de cada cliente aderir a produtos de investimento**.

Esse tipo de modelo é amplamente usado em bancos para:
- campanhas de cross-sell
- recomendação de produtos
- priorização comercial

### 4. Tempo até Conversão (Survival Analysis)
O modelo de **survival analysis (Cox Proportional Hazards)** estima o tempo esperado até a adoção de investimento.

Isso permite prever:
- quando clientes tendem a converter
- quais fatores aceleram ou retardam a adoção
- quais segmentos possuem maior potencial de conversão no curto prazo

### Aplicações no mundo real

Esse tipo de pipeline pode ser utilizado por instituições financeiras para:

- segmentação de clientes
- otimização de campanhas de marketing
- recomendação de produtos financeiros
- análise de ciclo de vida do cliente
- planejamento estratégico baseado em dados

💼 Aplicação no Contexto Bancário

Essa arquitetura permite:

Segmentação automática de clientes
Priorização de ofertas personalizadas
Identificação de trajetórias de risco
Modelagem de tempo até adoção de produto

📌 Observação

Este projeto foi desenvolvido como estudo aplicado para demonstrar técnicas de IA e Machine Learning em contexto de produtos financeiros.




