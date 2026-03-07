# Pipeline do Projeto — Bank Product Usage Profiling (AI)

## 1. Geração de dados (src.generate_data)
- Objetivo: Gerar uma base sintética de transações mensais de clientes bancários, simulando atributos demográficos, comportamento financeiro, uso de produtos e adoção de investimento ao longo do tempo.
- Entradas:Não há entrada externa de arquivos. O script utiliza apenas parâmetros internos definidos na classe Config:
n_customers: número de clientes simulados
n_months: número de meses simulados
seed: semente aleatória para reprodutibilidade
out_path: caminho do arquivo de saída
- Saídas:Arquivo CSV em:
data/raw/transactions_monthly.csv
Esse arquivo contém um painel mensal de clientes, com uma linha por cliente por mês.
- Principais colunas:
customer_id: identificador único do cliente
age: idade do cliente
income: renda estimada do cliente
month: mês da observação
balance: saldo financeiro simulado no mês
card_spend: gasto com cartão no mês
pix_count: quantidade de transações PIX no mês
credit_limit: limite de crédito estimado
utilization: nível de utilização do crédito
late_payment: indicador de atraso de pagamento
uses_card: indicador de uso de cartão
uses_credit: indicador de uso de crédito
adopt_investment: indicador de adoção de investimento no mês
time_to_investment: tempo até adoção do investimento
event_investment: indica se houve adoção do investimento
first_adopt_month: primeiro mês em que o cliente adotou investimento

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
