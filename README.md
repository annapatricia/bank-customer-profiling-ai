# Bank Product Usage Profiling with AI

Projeto de portfólio demonstrando uma arquitetura de Inteligência Artificial para identificação de perfis de utilização de produtos bancários, combinando:

- Segmentação comportamental (não supervisionado)
- Dinâmica de perfis (Markov)
- Modelo de propensão por produto (supervisionado)
- Survival Analysis (tempo até adoção)

---

## 🎯 Objetivo

Demonstrar como técnicas de Machine Learning podem ser utilizadas para:

- Identificar perfis latentes de clientes
- Estimar probabilidade de adoção de produtos financeiros
- Modelar o tempo até contratação
- Apoiar estratégias de cross-sell e gestão de risco

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

4. **Markov (Transição de Perfis)** *(em construção)*

5. **Modelo de Propensão** *(em construção)*

6. **Survival Analysis** *(em construção)*

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

📁 Estrutura do Projeto

src/
data/
reports/
docs/

💼 Aplicação no Contexto Bancário

Essa arquitetura permite:

Segmentação automática de clientes

Priorização de ofertas personalizadas

Identificação de trajetórias de risco

Modelagem de tempo até adoção de produto

📌 Observação

Este projeto foi desenvolvido como estudo aplicado para demonstrar técnicas de IA e Machine Learning em contexto de produtos financeiros.

