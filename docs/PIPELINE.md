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



## 3. Clusterização (src.cluster_profiles)
- Objetivo:
- Pré-processamento:
- Modelo:
- Métrica:
- Saídas:
- Interpretação dos clusters:

## 4. Markov (src.markov_transitions)
...

## 5. Propensão (src.propensity_model)
...

## 6. Survival (src.survival_model)
...
