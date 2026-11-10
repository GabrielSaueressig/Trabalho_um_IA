import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

os.makedirs('img', exist_ok=True)

print('--- CARREGANDO DADOS DE REGRESSÃO (SEGMENTOS 500m) ---')
train_seg = pd.read_csv('dataSet/treinoTesteValidacaoSegmentos/treino_segmentos.csv')
val_seg = pd.read_csv('dataSet/treinoTesteValidacaoSegmentos/validacao_segmentos.csv')
test_seg = pd.read_csv('dataSet/treinoTesteValidacaoSegmentos/teste_segmentos.csv')

X_train = train_seg.drop(columns=['target_regressao', 'target_binario'])
y_train = train_seg['target_regressao']

X_val = val_seg.drop(columns=['target_regressao', 'target_binario'])
y_val = val_seg['target_regressao']

X_test = test_seg.drop(columns=['target_regressao', 'target_binario'])
y_test = test_seg['target_regressao']

print('\n--- Treinando MLPRegressor e Gerando Curva de Loss ---')
epocas = 200

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate_init=0.001,
    random_state=42,
)

train_loss, val_loss = [], []

for epoch in range(epocas):
  mlp_reg.partial_fit(X_train, y_train)

  train_loss.append(mlp_reg.loss_)

  # MSE na validação
  y_val_pred = mlp_reg.predict(X_val)
  mse_v = mean_squared_error(y_val, y_val_pred)
  val_loss.append(mse_v / 2.0)  # Dividido por 2 para parear com a loss_ da MLP


plt.figure(figsize=(8, 5))
plt.plot(
    range(1, epocas + 1),
    train_loss,
    label='Treino (MSE Loss)',
    color='blue',
    linewidth=2,
)
plt.plot(
    range(1, epocas + 1),
    val_loss,
    label='Validação (MSE Loss)',
    color='orange',
    linestyle='--',
    linewidth=2,
)
plt.title(
    'Curva de Aprendizado - MLP Regressão (Loss por Época)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Épocas')
plt.ylabel('Loss (Erro Quadrático Médio)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/curva_loss_regressao.png')
plt.show()

# Treinamento final da MLP no conjunto completo de Treino
mlp_final = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=200,
    random_state=42,
)
mlp_final.fit(X_train, y_train)
y_pred_mlp = mlp_final.predict(X_test)

# Garantir predições não negativas (não existem atropelamentos negativos)
y_pred_mlp = np.maximum(0, y_pred_mlp)


# Random Forest Regressor

print('--- Treinando Random Forest Regressor ---')
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
y_pred_rf = np.maximum(0, y_pred_rf)


# Métricas

def calcular_metricas_regressao(y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)
  return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}


df_resumo_regressao = pd.DataFrame([
    {
        'Modelo': 'MLP (Rede Neural)', **calcular_metricas_regressao(y_test, y_pred_mlp),
    },
    {
        'Modelo': 'Random Forest Regressor', **calcular_metricas_regressao(y_test, y_pred_rf),
    },
])


print(df_resumo_regressao.to_string(index=False))


# Gráfico 1: Valores Reais vs. Valores Preditos (MLP)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_mlp, alpha=0.6, color='blue', label='Predições MLP')
max_val = max(y_test.max(), y_pred_mlp.max())
plt.plot(
    [0, max_val],
    [0, max_val],
    'r--',
    linewidth=2,
    label='Linha Perfeita (y = x)',
)
plt.title(
    'Valores Reais × Valores Preditos (MLP Regressão)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Valores Reais (Atropelamentos Observados)')
plt.ylabel('Valores Preditos pela MLP')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/reais_vs_preditos_mlp.png')
plt.show()

# Gráfico 2: Gráfico de Resíduos (Resíduo = Real - Predito)
residuos_mlp = y_test - y_pred_mlp

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_mlp, residuos_mlp, alpha=0.6, color='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title(
    'Gráfico de Resíduos (MLP Regressão)', fontsize=12, fontweight='bold'
)
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos (Real - Predito)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/grafico_residuos_mlp.png')
plt.show()