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



🔥 Versão Melhorada do README
🏦 Bank Customer Profiling AI
Behavioral Segmentation + Propensity Modeling + Temporal Dynamics

Projeto demonstrando uma arquitetura completa de Machine Learning para análise comportamental de clientes bancários, combinando modelos supervisionados e não supervisionados com modelagem temporal.

O objetivo é simular um cenário real de banco digital ou instituição financeira que deseja:

Entender perfis comportamentais

Prever adoção de produtos

Estimar tempo até contratação

Monitorar risco dinâmico de migração de perfil

Construir um score integrado de priorização comercial

Dados sintéticos estruturados de forma realista para simular comportamento financeiro mensal.

🎯 Problema de Negócio

Como identificar:

Quais clientes têm maior potencial para investir?

Quem está próximo da decisão?

Quem pode migrar para um perfil de maior risco?

Como priorizar abordagem comercial com base em potencial + timing + risco?

🏗️ Arquitetura do Projeto

O pipeline combina quatro camadas analíticas complementares:

1️⃣ Segmentação Comportamental (Unsupervised)

K-Means com padronização

Identificação de perfis latentes

Avaliação via Silhouette Score

Geração de “profile cards”

Saída:

cluster_profile_cards.csv

cluster_summary.csv

2️⃣ Dinâmica Temporal (Markov Chains)

Modelagem da probabilidade de transição entre perfis comportamentais:

𝑃
(
𝑆
𝑡
𝑎
𝑡
𝑒
𝑡
+
1
∣
𝑆
𝑡
𝑎
𝑡
𝑒
𝑡
)
P(State
t+1
	​

∣State
t
	​

)

Permite:

Identificar estados mais estáveis

Estimar risco de migração para perfis de inadimplência

Calcular distribuição estacionária

Saída:

markov_transition_matrix.csv

markov_transition_counts.csv

3️⃣ Modelo de Propensão (Supervised ML)

Predição de:

𝑃
(
Investimento
∣
𝑃
𝑒
𝑟
𝑓
𝑖
𝑙
,
𝐹
𝑒
𝑎
𝑡
𝑢
𝑟
𝑒
𝑠
)
P(Investimento∣Perfil,Features)

Modelo:

XGBoost (ou Gradient Boosting fallback)

Métricas:

AUC

KS

Saída:

propensity_scores.csv

propensity_metrics.csv

4️⃣ Survival Analysis (Time-to-Event Modeling)

Modelagem do tempo até adoção de investimento usando Cox Proportional Hazards.

Permite estimar:

𝑃
(
Adotar at
e
ˊ
 
𝑡
)
P(Adotar at
e
ˊ
 t)

Probabilidade em 3 / 6 / 9 meses

Tempo esperado até contratação

Saída:

survival_probabilities.csv

survival_expected_time.csv

survival_cox_summary.csv

5️⃣ Score Integrado Final

Combinação de:

Propensão estrutural

Urgência temporal

Risco de migração comportamental

Resultado:

𝐹
𝑖
𝑛
𝑎
𝑙
 
𝑆
𝑐
𝑜
𝑟
𝑒
=
0.50
⋅
𝑃
𝑟
𝑜
𝑝
𝑒
𝑛
𝑠
𝑎
~
𝑜
+
0.30
⋅
𝑈
𝑟
𝑔
𝑒
^
𝑛
𝑐
𝑖
𝑎
+
0.20
⋅
𝑅
𝑖
𝑠
𝑐
𝑜
Final Score=0.50⋅Propens
a
~
o+0.30⋅Urg
e
^
ncia+0.20⋅Risco

Saída:

final_scores.csv

Esse score permite priorização estratégica de clientes para campanhas comerciais ou gestão preventiva.

📊 Principais Insights

Perfis convergem estruturalmente para estados intermediários.

Estados extremos apresentam menor estabilidade temporal.

Propensão isolada não captura timing.

Survival adiciona dimensão temporal.

Markov adiciona dimensão dinâmica de risco.

Integração das três camadas gera visão 360° do cliente.

🧠 Técnicas Utilizadas

Python

Pandas / NumPy

Scikit-learn

XGBoost

K-Means

StandardScaler

Cox Proportional Hazards (lifelines)

Cadeias de Markov

Engenharia de atributos comportamentais

🚀 Como Executar
pip install -r requirements.txt
python -m src.generate_data
python -m src.build_features
python -m src.cluster_profiles
python -m src.markov_transitions
python -m src.propensity_model
python -m src.survival_model
python -m src.score_final
📁 Estrutura
src/
  generate_data.py
  build_features.py
  cluster_profiles.py
  markov_transitions.py
  propensity_model.py
  survival_model.py
  score_final.py

data/
reports/
models/
💼 Aplicação Real no Contexto Bancário

Esta arquitetura pode ser aplicada em:

Bancos digitais

Fintechs

Área de CRM analítico

Crédito e cross-sell

Gestão de ciclo de vida do cliente

Permite:

Segmentação automática

Priorização comercial

Gestão de risco comportamental

Planejamento de campanhas orientadas por IA

🧩 Próximo Nível (Evoluções Possíveis)

Random Survival Forest

LightGBM com calibração

API em FastAPI

Dashboard em Streamlit

Feature importance interpretável (SHAP)

📌 Conclusão Estratégica

O projeto demonstra como combinar:

Segmentação comportamental

Modelagem preditiva

Modelagem temporal

Dinâmica probabilística

Para criar uma arquitetura integrada de IA aplicada a produtos financeiros.
