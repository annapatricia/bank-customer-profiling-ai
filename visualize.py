import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/transactions_monthly.csv')

print(df.shape)
print(df.head())

monthly_balance = df.groupby('month')['balance'].mean()
monthly_balance.plot(title='Saldo Medio por Mes')
plt.show()