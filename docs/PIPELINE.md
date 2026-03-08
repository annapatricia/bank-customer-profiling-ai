# Pipeline do Projeto — Bank Product Usage Profiling (AI)

## 1. Geração de dados (`src/generate_data`) 

### Objetivo
Gerar uma base **sintética de transações mensais de clientes bancários**, simulando:
- atributos demográficos
- comportamento financeiro
- uso de produtos bancários
- adoção de investimento ao longo do tempo

Essa base é utilizada para experimentos de **análise de dados, modelagem preditiva e simulação de comportamento de clientes**.

---

### Entradas
O script **não utiliza arquivos externos**.  
Todos os parâmetros são definidos internamente na classe `Config`:

- `n_customers` — número de clientes simulados  
- `n_months` — número de meses simulados  
- `seed` — semente aleatória para garantir reprodutibilidade  
- `out_path` — caminho do arquivo de saída  

---

### Saída
O script gera um arquivo CSV em:
data/raw/transactions_monthly.csv


Esse arquivo representa um **painel mensal de clientes**, contendo **uma linha por cliente por mês**.

---

### Principais colunas

| Coluna | Descrição |
|------|------|
| `customer_id` | Identificador único do cliente |
| `age` | Idade do cliente |
| `income` | Renda estimada do cliente |
| `month` | Mês da observação |
| `balance` | Saldo financeiro simulado no mês |
| `card_spend` | Gasto com cartão no mês |
| `pix_count` | Número de transações PIX no mês |
| `credit_limit` | Limite de crédito estimado |
| `utilization` | Taxa de utilização do crédito |
| `late_payment` | Indicador de atraso de pagamento |
| `uses_card` | Indicador de uso de cartão |
| `uses_credit` | Indicador de uso de crédito |
| `adopt_investment` | Indicador de adoção de investimento no mês |
| `time_to_investment` | Tempo até adoção do investimento |
| `event_investment` | Indica se o cliente adotou investimento |
| `first_adopt_month` | Primeiro mês em que ocorreu a adoção |


## 2. Feature Engineering (`src/build_features`)

### Objetivo
Transformar os dados transacionais mensais em um **conjunto de features agregadas por cliente**, adequadas para análises estatísticas e modelos de machine learning.

Essa etapa resume o comportamento financeiro de cada cliente ao longo do tempo, criando variáveis que representam padrões médios de uso de produtos e risco financeiro.

---

### Entradas
Arquivo gerado na etapa anterior:
data/raw/transactions_monthly.csv

Contém um **painel mensal de clientes**, com múltiplas observações por cliente.

---

### Saída

Arquivo processado:
data/processed/customer_features.csv

Cada linha representa **um cliente**, contendo features agregadas que descrevem seu comportamento financeiro ao longo do período observado.

---

### Features criadas

| Feature | Descrição |
|------|------|
| `customer_id` | Identificador único do cliente |
| `age` | Idade do cliente |
| `income` | Renda estimada do cliente |
| `mean_balance` | Saldo médio ao longo dos meses |
| `std_balance` | Variabilidade do saldo ao longo do tempo |
| `mean_card_spend` | Gasto médio com cartão |
| `mean_utilization` | Taxa média de utilização do crédito |
| `mean_pix` | Número médio de transações PIX |
| `late_payment_rate` | Proporção de meses com atraso de pagamento |
| `adopted_ever` | Indicador se o cliente adotou investimento em algum momento |
| `time_to_investment` | Tempo até adoção do investimento (em meses) |



## 3. Clusterização (`src/cluster_profiles`)

### Objetivo
Segmentar os clientes em **perfis comportamentais** com base em características financeiras e transacionais, permitindo identificar grupos com padrões semelhantes de renda, saldo, uso de crédito, comportamento digital e risco.

Essa etapa apoia análises de **segmentação de clientes**, **estratégias de cross-sell/up-sell** e priorização comercial.

---

### Pré-processamento
O script utiliza como entrada o dataset de features agregadas por cliente:

```text
data/processed/customer_features.csv

As variáveis selecionadas para clusterização são:

| Feature           | Descrição                                  |
| ----------------- | ------------------------------------------ |
| age               | Idade do cliente                           |
| income            | Renda estimada do cliente                  |
| mean_balance      | Saldo médio ao longo do período            |
| std_balance       | Variabilidade do saldo                     |
| mean_card_spend   | Gasto médio com cartão                     |
| mean_utilization  | Utilização média do crédito                |
| mean_pix          | Número médio de transações PIX             |
| late_payment_rate | Proporção de meses com atraso de pagamento |
```

Antes do treinamento, essas features passam por padronização com StandardScaler, etapa essencial para métodos baseados em distância, como o K-Means, garantindo que variáveis em escalas diferentes não dominem a formação dos grupos.

---

### Modelo

Foi utilizado o algoritmo K-Means, com os seguintes parâmetros:

| Parâmetro    | Valor | Descrição                             |
| ------------ | ----- | ------------------------------------- |
| n_clusters   | 4     | Número de clusters do K-Means         |
| n_init       | 20    | Número de inicializações do algoritmo |
| random_state | 42    | Semente para reprodutibilidade        |

O modelo divide os clientes em 4 clusters, atribuindo cada cliente ao grupo cujo centroide seja mais próximo no espaço das features padronizadas.

### Métrica

A qualidade da clusterização é avaliada pela métrica Silhouette Score, calculada sobre os dados padronizados.

Essa métrica mede o quanto os clientes estão:

-próximos do próprio cluster
-distantes dos demais clusters

Quanto maior o valor, melhor a separação entre os grupos.

### Saídas

O script gera os seguintes arquivos:

Dados processados

data/processed/customer_features_with_cluster.csv
Dataset de clientes com o rótulo numérico do cluster

data/processed/customer_features_with_cluster_named.csv
Dataset de clientes com o rótulo numérico e o nome interpretável do cluster

### Tabelas de resumo

reports/tables/cluster_summary.csv
Médias das variáveis por cluster

reports/tables/cluster_profile_cards.csv
Tabela com nome do perfil, quantidade de clientes e descrição interpretativa

### Relatórios em Markdown

reports/cluster_report.md
Relatório resumido da clusterização, incluindo o valor de silhouette e a tabela de médias por cluster

reports/cluster_profile_cards.md
Cartões descritivos dos perfis identificados

### Interpretação dos clusters

Após a clusterização, os grupos recebem nomes descritivos com base em heurísticas simples sobre renda, risco, uso digital e uso de crédito.

Os perfis possíveis são:

Digital Crédito Intensivo
Alto uso digital (PIX) e maior risco de atraso. Perfil mais sensível a ações de gestão de crédito e prevenção.

Alta Renda Estável
Clientes com maior renda e menor risco. Perfil com potencial para investimentos e ofertas premium.

Digital Estável
Alto uso digital combinado com baixa inadimplência. Perfil promissor para expansão de portfólio, como investimentos e seguros.

Conservador Tradicional
Uso digital mais moderado e baixo risco. Perfil mais tradicional, com boa aderência a produtos simples e ações de educação financeira.

### Resultado esperado

Ao final da etapa, cada cliente passa a ter um perfil comportamental identificado, facilitando:

-segmentação de base
-personalização de ofertas
-análise estratégica de clientes
-apoio à tomada de decisão orientada por dados

## 4. Cadeias de Markov (`src/markov_transitions`)

### Objetivo
Modelar a **dinâmica de transição entre perfis de clientes ao longo do tempo**, utilizando uma **cadeia de Markov** baseada nos clusters identificados anteriormente.

Essa etapa permite analisar **como os clientes evoluem entre diferentes perfis comportamentais mês a mês**, identificando padrões de estabilidade ou mudança entre segmentos.

---

### Entradas
O script utiliza como entrada o dataset transacional mensal gerado na etapa de geração de dados:
data/raw/transactions_monthly.csv


Esse dataset contém **uma linha por cliente por mês**, incluindo informações de saldo, gastos, utilização de crédito e comportamento digital.

Caso ainda não exista um modelo de cluster salvo, o script também utiliza:
data/raw/transactions_monthly.csv

Esse dataset contém **uma linha por cliente por mês**, incluindo informações de saldo, gastos, utilização de crédito e comportamento digital.
data/processed/customer_features.csv


para treinar o modelo **K-Means** e o **StandardScaler**, que podem ser armazenados em:
models/scaler.pkl
models/kmeans.pkl


---

### Etapas do processamento

#### 1. Construção da tabela mensal
Os dados brutos são organizados em uma tabela contendo:

- `customer_id`
- `month`
- `age`
- `income`
- `balance`
- `card_spend`
- `utilization`
- `pix_count`
- `late_payment`

Cada linha representa **o estado financeiro de um cliente em determinado mês**.

---

#### 2. Atribuição de clusters por mês
O modelo **K-Means** previamente treinado é aplicado aos dados mensais.

Como o modelo foi treinado em features agregadas, é utilizada uma **aproximação compatível**, onde métricas mensais são utilizadas como proxies:

| Feature do modelo | Variável mensal utilizada |
|------|------|
| `mean_balance` | `balance` |
| `std_balance` | 0 (aproximação) |
| `mean_card_spend` | `card_spend` |
| `mean_utilization` | `utilization` |
| `mean_pix` | `pix_count` |
| `late_payment_rate` | `late_payment` |

O resultado é um dataset onde **cada cliente possui um cluster em cada mês**.

Arquivo gerado:
data/processed/customer_monthly_with_cluster.csv


---

### Modelo
A dinâmica dos clusters é modelada utilizando **Cadeias de Markov de primeira ordem**, onde a probabilidade de transição depende apenas do estado atual.

Formalmente:

\[
P(C_{t+1} | C_t)
\]

ou seja, a probabilidade de um cliente estar no cluster \(C_{t+1}\) dado que estava no cluster \(C_t\).

---

### Saídas

#### Tabela de transições (contagem)
reports/tables/markov_transition_counts.csv


Contém o número de transições observadas entre clusters.

Exemplo:

| state | next_state | count |
|------|------|------|
| 0 | 0 | ... |
| 0 | 1 | ... |
| 1 | 2 | ... |

---

#### Matriz de probabilidades de transição
reports/tables/markov_transition_matrix.csv


Cada linha representa um **estado atual (cluster)** e cada coluna o **estado seguinte**, contendo as probabilidades:

\[
P(cluster_{t+1} | cluster_t)
\]

---

### Interpretação
A matriz de transição permite identificar:

- **estabilidade dos clusters**  
  (clientes que permanecem no mesmo perfil)

- **migração entre perfis**  
  (ex: clientes que evoluem de perfil tradicional para digital)

- **trajetórias de comportamento financeiro**

Essas informações podem apoiar análises de:

- ciclo de vida do cliente
- evolução de comportamento financeiro
- impacto de estratégias comerciais
- previsão de mudanças de perfil


## 5. Propensão (`src/propensity_model`)

### Objetivo
Treinar um modelo de **propensão à adoção de investimento**, estimando a probabilidade de cada cliente aderir a um produto de investimento com base em seu perfil financeiro, comportamento transacional e cluster comportamental.

Essa etapa apoia ações de:
- segmentação comercial
- priorização de clientes
- campanhas de cross-sell
- recomendação de produtos financeiros

---

### Entradas
O script utiliza como base principal o dataset de features com cluster já atribuído.

Prioridade de leitura:

```text
data/processed/customer_features_with_cluster_named.csv
```
uma opção alternativa seria:
ou, caso o arquivo acima não exista:
Esses arquivos devem ser gerados previamente nas etapas:

src/build_features

src/cluster_profiles

### Target

A variável alvo do modelo é:

adopted_ever — indica se o cliente adotou investimento em algum momento do período observado

Trata-se de um problema de classificação binária:

1 = cliente adotou investimento

0 = cliente não adotou investimento

### Features utilizadas

O modelo utiliza variáveis numéricas relacionadas ao perfil do cliente, comportamento financeiro e cluster.

Variáveis numéricas

| Feature               | Tipo                  | Descrição                                           |
| --------------------- | --------------------- | --------------------------------------------------- |
| age                   | Numérica              | Idade do cliente                                    |
| income                | Numérica              | Renda estimada do cliente                           |
| m12_mean_balance      | Numérica              | Saldo médio dos últimos 12 meses                    |
| m12_std_balance       | Numérica              | Variabilidade do saldo nos últimos 12 meses         |
| m12_mean_card_spend   | Numérica              | Gasto médio com cartão nos últimos 12 meses         |
| m12_mean_utilization  | Numérica              | Utilização média do crédito nos últimos 12 meses    |
| m12_mean_pix          | Numérica              | Número médio de transações PIX nos últimos 12 meses |
| m12_late_payment_rate | Numérica              | Proporção de atrasos nos últimos 12 meses           |
| m3_mean_balance       | Numérica              | Saldo médio dos últimos 3 meses                     |
| m3_std_balance        | Numérica              | Variabilidade do saldo nos últimos 3 meses          |
| m3_mean_card_spend    | Numérica              | Gasto médio com cartão nos últimos 3 meses          |
| m3_mean_utilization   | Numérica              | Utilização média do crédito nos últimos 3 meses     |
| m3_mean_pix           | Numérica              | Número médio de transações PIX nos últimos 3 meses  |
| m3_late_payment_rate  | Numérica              | Proporção de atrasos nos últimos 3 meses            |
| cluster               | Numérica              | Identificador do cluster do cliente                 |
| cluster_name          | Categórica (opcional) | Nome interpretável do perfil de cluster             |


Observação: o script utiliza apenas as colunas que estiverem disponíveis no dataset, mantendo flexibilidade para diferentes versões das features

Pré-processamento

Antes do treinamento, os dados passam por um pipeline de preparação:

Variáveis numéricas

imputação de valores ausentes com a mediana

Variáveis categóricas

imputação com o valor mais frequente

codificação com One-Hot Encoding

Esse pré-processamento é implementado com ColumnTransformer e Pipeline, garantindo consistência entre treino e inferência.

Modelo

O script utiliza preferencialmente o modelo XGBoost Classifier, caso a biblioteca esteja instalada.

Parâmetros principais:

n_estimators = 300

max_depth = 4

learning_rate = 0.05

subsample = 0.9

colsample_bytree = 0.9

random_state = 42

Caso o XGBoost não esteja disponível, o script utiliza como alternativa:

HistGradientBoostingClassifier

Essa abordagem garante que o pipeline funcione mesmo sem dependências extras.

Validação

Os dados são divididos em:

75% treino

25% teste

A separação é feita com train_test_split, usando:

random_state = 42

stratify = y

O uso de stratify preserva a proporção entre classes no treino e no teste.

Métricas

O desempenho do modelo é avaliado por duas métricas principais:

AUC (ROC AUC)

Mede a capacidade do modelo de distinguir clientes que adotam investimento daqueles que não adotam.

KS (Kolmogorov-Smirnov)

Mede a separação entre as distribuições de score das classes positiva e negativa, sendo bastante utilizado em contexto bancário e de risco.

Saídas
Métricas do modelo
reports/tables/propensity_metrics.csv

Contém:

auc

ks

model

Scores de propensão por cliente
reports/tables/propensity_scores.csv

Contém:

customer_id

propensity_investment

Cada linha representa a probabilidade prevista de o cliente adotar investimento.

Modelo treinado
models/propensity_model.pkl

O modelo é salvo com joblib quando a biblioteca está disponível.

Interpretação

O score de propensão permite identificar:

clientes com maior chance de aderir a investimento

segmentos prioritários para campanhas comerciais

oportunidades de cross-sell

perfis com maior potencial de conversão

Na prática, clientes com maior score podem ser priorizados em ações de relacionamento, recomendação de produtos ou estratégias de oferta personalizada.

Resultado esperado

Ao final da etapa, o projeto passa a contar com um modelo supervisionado capaz de:

prever propensão de investimento

ranquear clientes por potencial de conversão

apoiar decisões orientadas por dados em contexto bancário


Tem um detalhe importante: **esse texto assume que seu `build_features` já gera colunas como `m12_*` e `m3_*`**.  
Se o seu arquivo atual ainda estiver com `mean_balance`, `mean_pix`, etc., então o README e o código precisam ficar alinhados.

## 6. Survival (`src/survival_model`)

### Objetivo
Modelar o **tempo até a adoção de investimento** por cliente, utilizando análise de sobrevivência.

Essa etapa permite estimar:
- a probabilidade de um cliente **adotar investimento até determinados horizontes de tempo**
- o **tempo esperado até adoção**
- o efeito das variáveis explicativas sobre a velocidade de conversão

É uma abordagem útil quando o interesse não é apenas saber **se** o cliente vai adotar, mas também **quando** isso tende a acontecer.

---

### Entradas
O script utiliza como base principal a tabela de features por cliente.

Prioridade de leitura:

```text
data/processed/customer_features_with_cluster_named.csv

ou, se não existir:
data/processed/customer_features_with_cluster.csv

ou, como fallback:
data/processed/customer_features.csv

Esses arquivos devem ter sido gerados previamente nas etapas de:

src/build_features
src/cluster_profiles

Variável de sobrevivência
Duração
time_to_investment — representa o tempo até adoção do investimento, em meses

Evento
adopted_ever — indica se o cliente adotou investimento durante a janela observada

Interpretação:

1 = evento observado (cliente adotou investimento)
0 = observação censurada (cliente não adotou no período disponível)

Covariáveis utilizadas

O modelo utiliza as covariáveis disponíveis no dataset, priorizando variáveis comportamentais, financeiras e de cluster.

Possíveis variáveis:

age
income
m12_mean_balance
m12_std_balance
m12_mean_card_spend
m12_mean_utilization
m12_mean_pix
m12_late_payment_rate
m3_mean_balance
m3_std_balance
m3_mean_card_spend
m3_mean_utilization
m3_mean_pix
m3_late_payment_rate
cluster

O script utiliza apenas as colunas que estiverem presentes na base.

Pré-processamento

Antes do ajuste do modelo:
as colunas de duração e evento são convertidas para formato numérico
as covariáveis também são convertidas para tipo numérico
linhas com duração ausente são removidas
valores ausentes nas covariáveis são preenchidos com a mediana

Esse tratamento garante consistência para o ajuste do modelo de sobrevivência.

Modelo

O método utilizado é o Cox Proportional Hazards Model (CoxPHFitter), da biblioteca lifelines.

Parâmetro utilizado:
penalizer = 0.01

O modelo de Cox estima o efeito das variáveis sobre a taxa de risco (hazard) de adoção de investimento ao longo do tempo.

De forma intuitiva:
coeficientes positivos indicam maior chance de adoção mais cedo
coeficientes negativos indicam menor chance de adoção no curto prazo

Previsões geradas
Probabilidade de adoção até horizontes específicos
O script calcula a probabilidade acumulada de adoção até:
3 meses
6 meses
9 meses

A partir da função de sobrevivência 𝑆(𝑡), a probabilidade de adoção até o tempo 
𝑡 é dada por:

𝑃(𝑇≤𝑡)=1−𝑆(𝑡)
P(T≤t)=1−S(t)

Tempo esperado até adoção
O script também estima o tempo esperado até adoção, aproximando:

𝐸[𝑇]≈∑𝑡𝑆(𝑡)
E[T]≈t∑S(t)

Essa métrica resume, em meses, quanto tempo o cliente tende a levar para adotar investimento.

Saídas
Probabilidades de adoção por horizonte
reports/tables/survival_probabilities.csv

Contém:
customer_id
p_adopt_3m
p_adopt_6m
p_adopt_9m

Tempo esperado até adoção
reports/tables/survival_expected_time.csv

Contém:
customer_id
expected_time_months

Resumo dos coeficientes do modelo de Cox
reports/tables/survival_cox_summary.csv

Contém estatísticas do modelo, incluindo:
coeficientes
erro padrão
estatística de teste
p-valor

Relatório em Markdown
reports/survival_report.md

Traz um resumo do modelo, covariáveis utilizadas e os coeficientes mais relevantes.

Modelo salvo
models/survival_cox.pkl

O modelo é salvo em formato pickle quando a biblioteca joblib está disponível.

Interpretação
Essa etapa permite responder perguntas como:
-quais clientes tendem a adotar investimento mais rapidamente
-quais variáveis aceleram ou retardam a conversão
-qual a probabilidade de adoção em diferentes horizontes de tempo
-quais perfis apresentam maior potencial de conversão no curto prazo
Em contexto de negócio, isso pode ser usado para:
-planejamento de campanhas por janela temporal
-priorização comercial
-definição de estratégias de relacionamento
-análise de ciclo de vida do cliente

Resultado esperado
Ao final da etapa, o projeto passa a contar com um modelo capaz de analisar tempo até conversão, complementando o modelo de propensão com uma visão temporal mais rica do comportamento de adoção de investimento.



...
