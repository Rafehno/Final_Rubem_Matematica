import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('AirQuality.csv', delimiter=';')

data = data.replace({',': '.'}, regex=True) #substitui as "," por "." para evitar erros em calculos 

data.isnull().sum() # verificiar e tratar dados que estejam faltando

data = data.loc[:, ~data.columns.str.contains('^Unnamed')] # remover colunas sem nome

data.columns = data.columns.str.strip() # colocar os dados faltantes como médias

def calcular_iqa(row):
    try:
        co = float(row['CO(GT)'])  # Converte para float
        no2 = float(row['NO2(GT)'])  
        o3 = float(row['PT08.S5(O3)'])  
        pm10 = float(row['PT08.S1(CO)'])  
        so2 = float(row['PT08.S4(NO2)']) 
    except ValueError:
        
        return float('nan') # se tiver erro ficará como nan
    
    def calcular_qualidade(poluente, limite_baixo, limite_alto):
        if poluente < limite_baixo:
            return 100  # IQA alto (excelente)
        elif poluente > limite_alto:
            return 0    # IQA baixo (ruim)
        else:
            return (100 * (limite_alto - poluente)) / (limite_alto - limite_baixo)

    # IQA para cada poluente
    qualidade_co = calcular_qualidade(co, 0, 5)
    qualidade_no2 = calcular_qualidade(no2, 0, 200)
    qualidade_o3 = calcular_qualidade(o3, 0, 100)
    qualidade_pm10 = calcular_qualidade(pm10, 0, 50)
    qualidade_so2 = calcular_qualidade(so2, 0, 150)

    # Pesos
    w_co = 0.3
    w_no2 = 0.3
    w_o3 = 0.2
    w_pm10 = 0.1
    w_so2 = 0.1

    # IQA calculo
    iqa = (qualidade_co * w_co + qualidade_no2 * w_no2 + qualidade_o3 * w_o3 + qualidade_pm10 * w_pm10 + qualidade_so2 * w_so2)

    iqa = max(0, min(100, iqa))

    return iqa

# mostra as primeiras linhas do dataset, a fim de evitar erros (comentada no momento)
# print(data.head())

data['IQA'] = data.apply(calcular_iqa, axis=1) #calcular IQA

# definir x e y
X = data[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']]

y = data['IQA']  

# treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalização de dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinr o modelo de regressão linear
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# plotar o gráfico com resultados
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Valores Reais de IQA')
plt.ylabel('Valores Preditos de IQA')
plt.title('Comparação entre os valores reais e preditos de IQA')
plt.show()

# calcular residuos para poder plotar eles separadamente
residuos = y_test - y_pred

# plotar os resíduos em gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores Preditos de IQA')
plt.ylabel('Resíduos (Diferença entre Real e Predito)')
plt.title('Gráfico de Resíduos')
plt.show()

# salvar os resultados em um TXT
def salvar_resultados_no_txt(mae, mse, rmse):
    output_file = 'resultados_modelo.txt'
    # verificar se existe o txt
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
           # criar um cabeçalho
           f.write('Modelo de Regressão Linear - Resultados de Avaliação\n')
           f.write('---------------------------------------------\n')

    # colocar os resultados no txt
    with open(output_file, 'a') as f:
       f.write(f'MAE: {mae}\n')
       f.write(f'MSE: {mse}\n')
       f.write(f'RMSE: {rmse}\n')
       f.write('---------------------------------------------\n')

# salvar_resultados_no_txt(mae, mse, rmse)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # para comparação, mas ignorar momentaneamente
# plt.show() 

# ------------------ teste com RandomForestRegressor (mais confiável)

X = data[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']]
y = data['IQA']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento com o RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Avaliar o modelo com cross-validation
scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
mean_score = -scores.mean()  # Negativo porque cross_val_score usa valores negativos para erro absoluto

print(f'-- Cross-Validation MAE: {mean_score}')

# Fazer previsões
y_pred_rf = rf_model.predict(X_test_scaled)

# Avaliar o modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f'Mean Absolute Error (RandomForest): {mae_rf}')
print(f'Mean Squared Error (RandomForest): {mse_rf}')
print(f'Root Mean Squared Error (RandomForest): {rmse_rf}')

def salvar_resultados_rf_no_txt(mae, mse, rmse):
    output_file = 'resultados_randomforest.txt'

    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
           
            f.write('Modelo RandomForestRegressor - Resultados de Avaliação\n')
            f.write('---------------------------------------------\n')

    
    with open(output_file, 'a') as f:
        f.write(f'MAE: {mae}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'RMSE: {rmse}\n')
        f.write('---------------------------------------------\n')

salvar_resultados_rf_no_txt(mae_rf, mse_rf, rmse_rf)

# Plotar em gráfico os resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Valores Reais de IQA')
plt.ylabel('Valores Preditos de IQA')
plt.title('Comparação entre os valores reais e preditos de IQA - RandomForest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # linha de identidade
plt.show()

# Calcular os resíduos (desta vez com o RandomForest)
residuos_rf = y_test - y_pred_rf

# Plotar os resíduos em gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_rf, y=residuos_rf)
plt.axhline(0, color='red', linestyle='--')  # Linha para marcar o "zero" dos resíduos
plt.xlabel('Valores Preditos de IQA')
plt.ylabel('Resíduos (Diferença entre Real e Predito)')
plt.title('Gráfico de Resíduos - RandomForestRegressor')
plt.show()
