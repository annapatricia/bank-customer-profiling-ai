# Bank Product Usage Profiling with AI
#Clusterização + Propensão + Dinâmica Temporal (Markov)

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

