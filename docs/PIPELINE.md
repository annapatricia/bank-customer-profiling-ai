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


## 2. Feature engineering (src.build_features)
- Objetivo:
- Entradas:
- Saídas:
- Features criadas:

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
