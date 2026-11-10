import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

os.makedirs('img', exist_ok=True)

# 1. Carregar os CSVs dos Segmentos de 500m
print('--- CARREGANDO DADOS DOS SEGMENTOS (500m) ---')
train_seg = pd.read_csv(
    'dataSet/treinoTesteValidacaoSegmentos/treino_segmentos.csv'
)
val_seg = pd.read_csv(
    'dataSet/treinoTesteValidacaoSegmentos/validacao_segmentos.csv'
)
test_seg = pd.read_csv(
    'dataSet/treinoTesteValidacaoSegmentos/teste_segmentos.csv'
)

X_train = train_seg.drop(columns=['target_regressao', 'target_binario'])
y_train = train_seg['target_binario']

X_val = val_seg.drop(columns=['target_regressao', 'target_binario'])
y_val = val_seg['target_binario']

X_test = test_seg.drop(columns=['target_regressao', 'target_binario'])
y_test = test_seg['target_binario']

pesos_train = compute_sample_weight('balanced', y_train)

# ==============================================================================
# 2. TREINAMENTO DA MLP BINÁRIA
# ==============================================================================
print('\n--- Treinando MLP Binária ---')
mlp_bin = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate_init=0.0005,
    max_iter=400,
    batch_size=200,
    random_state=42,
)
mlp_bin.fit(X_train, y_train, sample_weight=pesos_train)

y_pred_mlp = mlp_bin.predict(X_test)
y_probs_mlp = mlp_bin.predict_proba(X_test)[:, 1]

# ==============================================================================
# 3. TREINAMENTO DA RANDOM FOREST BINÁRIA (MODELO TRADICIONAL)
# ==============================================================================
print('--- Treinando Random Forest Binária ---')
rf_bin = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
rf_bin.fit(X_train, y_train)

y_pred_rf = rf_bin.predict(X_test)
y_probs_rf = rf_bin.predict_proba(X_test)[:, 1]


# ==============================================================================
# 4. CÁLCULO DAS MÉTRICAS EXIGIDAS
# ==============================================================================
def calcular_metricas_binarias(y_true, y_pred, y_probs):
  return {
      'Accuracy': accuracy_score(y_true, y_pred),
      'Precision': precision_score(y_true, y_pred),
      'Recall': recall_score(y_true, y_pred),
      'F1-Score': f1_score(y_true, y_pred),
      'ROC-AUC': roc_auc_score(y_true, y_probs),
  }


df_resumo_binario = pd.DataFrame([
    {
        'Modelo': 'MLP (Rede Neural)',
        **calcular_metricas_binarias(y_test, y_pred_mlp, y_probs_mlp),
    },
    {
        'Modelo': 'Random Forest',
        **calcular_metricas_binarias(y_test, y_pred_rf, y_probs_rf),
    },
])

print('\n===============================================================')
print('  RESULTADOS BINÁRIOS: ALTO RISCO (1) VS. BAIXO RISCO (0)      ')
print('===============================================================')
print(df_resumo_binario.to_string(index=False))

# ==============================================================================
# 5. PLOTAR E SALVAR A CURVA ROC E MATRIZ DE CONFUSÃO
# ==============================================================================
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_probs_mlp)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr_mlp,
    tpr_mlp,
    label=f"MLP (AUC = {df_resumo_binario.loc[0, 'ROC-AUC']:.4f})",
    color='blue',
    linewidth=2,
)
plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {df_resumo_binario.loc[1, 'ROC-AUC']:.4f})",
    color='green',
    linewidth=2,
)
plt.plot([0, 1], [0, 1], 'k--', label='Chute Aleatório (AUC = 0.5000)')
plt.title(
    'Curva ROC - Classificação Binária de Hotspots (500m)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
plt.ylabel('Taxa de Verdadeiros Positivos (Recall / Sensibilidade)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/curva_roc_binaria.png')
plt.show()

# Matriz de Confusão da MLP Binária
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_mlp),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Baixo Risco', 'Alto Risco'],
    yticklabels=['Baixo Risco', 'Alto Risco'],
)
plt.title(
    'Matriz de Confusão - MLP Binária (Hotspots)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.savefig('img/matriz_confusao_binaria.png')
plt.show()