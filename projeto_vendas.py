import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. Carregar os dados de vendas de pizzas e bebidas
pizza_data = pd.read_csv('relatorio_pizza.csv')
bebidas_data = pd.read_csv('relatorio_bebidas.csv')

# Garantir que haja uma coluna de 'quantidade'
# Para cada linha, presumimos que um item foi vendido por registro (transação)
pizza_data['quantidade'] = 1
bebidas_data['quantidade'] = 1

# 2. Analisar a quantidade e valor vendidos por produto

# Agrupar as vendas de pizza por produto (quantidade de itens vendidos e valor total)
vendas_pizza = pizza_data.groupby('produto').agg(quantidade=('produto', 'size'), valor_total=('valor', 'sum')).reset_index()

# Agrupar as vendas de bebidas por produto (quantidade de itens vendidos e valor total)
vendas_bebidas = bebidas_data.groupby('produto').agg(quantidade=('produto', 'size'), valor_total=('valor', 'sum')).reset_index()

# Exibir os resultados da análise
print("\nQuantidade e Valor Vendido de Pizzas por Produto:")
print(vendas_pizza)
print("\nQuantidade e Valor Vendido de Bebidas por Produto:")
print(vendas_bebidas)

# 3. Criar gráficos de vendas por produto (quantidade e valor)

# Gráfico de barras para as vendas de pizzas (quantidade)
plt.bar(vendas_pizza['produto'], vendas_pizza['quantidade'])
plt.title('Quantidade de Pizzas Vendidas')
plt.ylabel('Quantidade Vendida')
plt.xlabel('Produto')
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras para as vendas de pizzas (valor total)
plt.bar(vendas_pizza['produto'], vendas_pizza['valor_total'])
plt.title('Valor Total de Pizzas Vendidas')
plt.ylabel('Valor Total Vendido (R$)')
plt.xlabel('Produto')
plt.xticks(rotation=45)
plt.show()

# Gráfico de barras para as vendas de bebidas (quantidade)
plt.bar(vendas_bebidas['produto'], vendas_bebidas['quantidade'])
plt.title('Quantidade de Bebidas Vendidas')
plt.ylabel('Quantidade Vendida')
plt.xlabel('Produto')
plt.xticks(rotation=45)
plt.show()

# 4. Analisar as vendas por hora

# Converter a coluna de data para o formato datetime e extrair a hora
pizza_data['hora'] = pd.to_datetime(pizza_data['data'], format='%d/%m/%Y %H:%M:%S').dt.hour
bebidas_data['hora'] = pd.to_datetime(bebidas_data['data'], format='%d/%m/%Y %H:%M:%S').dt.hour

# Ajustar para considerar apenas o horário de funcionamento (das 18h às 2h)
pizza_data = pizza_data[(pizza_data['hora'] >= 18) | (pizza_data['hora'] <= 2)]
bebidas_data = bebidas_data[(bebidas_data['hora'] >= 18) | (bebidas_data['hora'] <= 2)]

# Agrupar as vendas por hora
vendas_por_hora_pizza = pizza_data.groupby('hora')['quantidade'].sum()
vendas_por_hora_bebidas = bebidas_data.groupby('hora')['quantidade'].sum()

# Exibir os resultados
print("\nVendas de Pizzas por Hora (Horário de Funcionamento):")
print(vendas_por_hora_pizza)
print("\nVendas de Bebidas por Hora (Horário de Funcionamento):")
print(vendas_por_hora_bebidas)

# 5. Criar gráficos de vendas por hora (somente horário de funcionamento)

# Gráfico de vendas por hora (pizzas)
plt.plot(vendas_por_hora_pizza.index, vendas_por_hora_pizza.values, marker='o')
plt.title('Vendas por Hora (Pizzas) - Horário de Funcionamento')
plt.ylabel('Quantidade Vendida')
plt.xlabel('Hora do Dia')
plt.xticks([18, 19, 20, 21, 22, 23, 0, 1, 2])  # Mostrar apenas as horas de funcionamento
plt.grid(True)
plt.show()

# Gráfico de vendas por hora (bebidas)
plt.plot(vendas_por_hora_bebidas.index, vendas_por_hora_bebidas.values, marker='o')
plt.title('Vendas por Hora (Bebidas) - Horário de Funcionamento')
plt.ylabel('Quantidade Vendida')
plt.xlabel('Hora do Dia')
plt.xticks([18, 19, 20, 21, 22, 23, 0, 1, 2])  # Mostrar apenas as horas de funcionamento
plt.grid(True)
plt.show()

# 6. Previsão de Vendas para os Próximos 7 Dias

# Previsão de quantidade de pizzas vendidas
pizza_data['data'] = pd.to_datetime(pizza_data['data'], format='%d/%m/%Y %H:%M:%S').dt.date
vendas_diarias_pizza = pizza_data.groupby('data')['quantidade'].sum()
model_pizza = ARIMA(vendas_diarias_pizza, order=(5,1,0))
model_pizza_fit = model_pizza.fit()
forecast_pizza = model_pizza_fit.forecast(steps=7)

# Exibir a previsão de pizzas
print("\nPrevisão de quantidade de pizzas vendidas para os próximos 7 dias:")
print(forecast_pizza)

# Previsão de quantidade e valor de bebidas
bebidas_data['data'] = pd.to_datetime(bebidas_data['data'], format='%d/%m/%Y %H:%M:%S').dt.date
vendas_diarias_bebidas = bebidas_data.groupby('data').agg(quantidade=('quantidade', 'sum'), valor_total=('valor', 'sum'))

# Previsão para quantidade de bebidas
model_bebidas_quantidade = ARIMA(vendas_diarias_bebidas['quantidade'], order=(5,1,0))
model_bebidas_quantidade_fit = model_bebidas_quantidade.fit()
forecast_bebidas_quantidade = model_bebidas_quantidade_fit.forecast(steps=7)

# Previsão para valor total de bebidas
model_bebidas_valor = ARIMA(vendas_diarias_bebidas['valor_total'], order=(5,1,0))
model_bebidas_valor_fit = model_bebidas_valor.fit()
forecast_bebidas_valor = model_bebidas_valor_fit.forecast(steps=7)

# Exibir as previsões de bebidas
print("\nPrevisão de quantidade de bebidas vendidas para os próximos 7 dias:")
print(forecast_bebidas_quantidade)
print("\nPrevisão de valor total de bebidas vendidas para os próximos 7 dias (R$):")
print(forecast_bebidas_valor)
