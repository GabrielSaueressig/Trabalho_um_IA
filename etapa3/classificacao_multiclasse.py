import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Carregar os CSVs da Estratégia A
train = pd.read_csv('dataSet/treinoTesteValidacao/treino.csv')
val = pd.read_csv('dataSet/treinoTesteValidacao/validacao.csv')
test = pd.read_csv('dataSet/treinoTesteValidacao/teste.csv')    

train_temp = pd.read_csv('dataSet/treinoTesteValidacao/treino_temporal.csv')
test_temp = pd.read_csv('dataSet/treinoTesteValidacao/teste_temporal.csv')

X_train = train.drop(columns=['target'])
y_train = train['target']

X_val = val.drop(columns=['target'])
y_val = val['target']

X_test = test.drop(columns=['target'])
y_test = test['target']

X_train_temp, y_train_temp = (
    train_temp.drop(columns=['target']),
    train_temp['target'],
)
X_test_temp, y_test_temp = (
    test_temp.drop(columns=['target']),
    test_temp['target'],
)


classes_nomes = ['Anfíbios', 'Aves', 'Mamíferos', 'Répteis']

# Grade de hiperparâmetros para testar no conjunto de Validação
# grades_config = [
#     {
#         'nome': 'Config 1 (64, 32 | alpha=0.01)',
#         'layers': (64, 32),
#         'alpha': 0.01,
#         'lr': 0.001,
#     },
#     {
#         'nome': 'Config 2 (128, 64 | alpha=0.01)',
#         'layers': (128, 64),
#         'alpha': 0.01,
#         'lr': 0.001,
#     },
#     {
#         'nome': 'Config 3 (128, 64, 32 | alpha=0.05)',
#         'layers': (128, 64, 32),
#         'alpha': 0.05,
#         'lr': 0.0005,
#     },
#     {
#         'nome': 'Config 4 (64, 64 | alpha=0.1)',
#         'layers': (64, 64),
#         'alpha': 0.1,
#         'lr': 0.001,
#     },
# ]

# melhor_f1 = 0
# melhor_modelo = None
# melhor_config_nome = ''

# print('\n--- Iniciando Busca de Hiperparâmetros no Conjunto de Validação ---')
# for config in grades_config:
#   mlp = MLPClassifier(
#       hidden_layer_sizes=config['layers'],
#       activation='relu',
#       solver='adam',
#       alpha=config['alpha'],
#       learning_rate_init=config['lr'],
#       max_iter=400,
#       batch_size=200,
#       random_state=42,
#   )


#   mlp.fit(X_train, y_train, sample_weight=pesos_amostra)

#   y_pred_val = mlp.predict(X_val)
#   f1_macro_val = f1_score(y_val, y_pred_val, average='macro')
#   acc_val = accuracy_score(y_val, y_pred_val)

#   print( f"-> {config['nome']}: F1-Macro Validação = {f1_macro_val:.4f} | Acurácia" f' = {acc_val:.4f}')

#   if f1_macro_val > melhor_f1:
#     melhor_f1 = f1_macro_val
#     melhor_modelo = mlp
#     melhor_config_nome = config['nome']


# Melhor arquitetura encontrada na validação: Config 3 (128, 64, 32 | alpha=0.05)
# DATASET ALEATORIO

pesos_train = compute_sample_weight('balanced', y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate_init=0.0005,
    max_iter=400,
    random_state=42,
    batch_size=200
)

mlp.fit(X_train, y_train, sample_weight=pesos_train)

y_pred_test = mlp.predict(X_test)
f1_test = f1_score(y_test, y_pred_test, average='macro')
acc_test = accuracy_score(y_test, y_pred_test)

print(f'Acurácia Global no Teste: {acc_test:.4f}')
print(f'F1-Score Macro no Teste:  {f1_test:.4f}\n')
print('Relatório de Classificação no Teste:')
print(classification_report(y_test, y_pred_test, target_names=classes_nomes))

plt.figure(figsize=(7, 5))
matriz_a = confusion_matrix(y_test, y_pred_test)
sns.heatmap(
    matriz_a,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes_nomes,
    yticklabels=classes_nomes,
)
plt.title(
    'Matriz de Confusão - Teste (Divisão Aleatória)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()

os.makedirs('img', exist_ok=True)

plt.savefig('img/matriz_confusao_aleatoria.png')
plt.show()

# DATASET COM DIVISAO TEMPORAL

print('\n=== EXPERIMENTO B: DIVISÃO TEMPORAL (TESTE 2014-2015) ===')
pesos_temp = compute_sample_weight('balanced', y_train_temp)

mlp_temporal = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate_init=0.0005,
    max_iter=400,
    random_state=42,
    batch_size=200
)

mlp_temporal.fit(X_train_temp, y_train_temp, sample_weight=pesos_temp)

# Predição no Teste Temporal
y_pred_temp = mlp_temporal.predict(X_test_temp)

f1_test_temp = f1_score(y_test_temp, y_pred_temp, average='macro')
acc_test_temp = accuracy_score(y_test_temp, y_pred_temp)

print(f'Acurácia Global no Teste Temporal: {acc_test_temp:.4f}')
print(f'F1-Score Macro no Teste Temporal:  {f1_test_temp:.4f}\n')
print('Relatório de Classificação Temporal (2014-2015):')
print(classification_report(y_test_temp, y_pred_temp, target_names=classes_nomes))

# Matriz de Confusão do Teste Temporal
plt.figure(figsize=(7, 5))
matriz_b = confusion_matrix(y_test_temp, y_pred_temp)
sns.heatmap(
    matriz_b,
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=classes_nomes,
    yticklabels=classes_nomes,
)
plt.title(
    'Matriz de Confusão - Teste (Divisão Temporal 2014-2015)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.savefig('img/matriz_confusao_temporal.png')
plt.show()


# Aprendizado Por Épocas

pesos_train = compute_sample_weight('balanced', y_train)

mlp_curva = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.05,
    learning_rate_init=0.0005,
    random_state=42,
)

epocas = 150
train_loss, val_loss = [], []
train_f1, val_f1 = [], []
classes_unicas = np.unique(y_train)

for epoch in range(epocas):
  mlp_curva.partial_fit(X_train, y_train, classes=classes_unicas, sample_weight=pesos_train)

  train_loss.append(mlp_curva.loss_)

  # stimativa da perda na Validação
  val_probs = mlp_curva.predict_proba(X_val)
  eps = 1e-15
  val_probs = np.clip(val_probs, eps, 1 - eps)
  val_onehot = pd.get_dummies(y_val).values
  loss_v = -np.mean(np.sum(val_onehot * np.log(val_probs), axis=1))
  val_loss.append(loss_v)

  # Guardar o F1-Macro de Treino e Validação
  train_f1.append(f1_score(y_train, mlp_curva.predict(X_train), average='macro'))
  val_f1.append(f1_score(y_val, mlp_curva.predict(X_val), average='macro'))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Loss
ax1.plot(range(1, epocas + 1), train_loss, label='Treino (Loss)', color='blue')
ax1.plot(
    range(1, epocas + 1),
    val_loss,
    label='Validação (Loss)',
    color='orange',
    linestyle='--',
)
ax1.set_title(
    'Evolução da Perda (Loss) por Época', fontsize=12, fontweight='bold'
)
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: F1-Score Macro
ax2.plot(
    range(1, epocas + 1), train_f1, label='Treino (F1 Macro)', color='green'
)
ax2.plot(
    range(1, epocas + 1),
    val_f1,
    label='Validação (F1 Macro)',
    color='red',
    linestyle='--',
)
ax2.set_title(
    'Evolução do F1-Score Macro por Época', fontsize=12, fontweight='bold'
)
ax2.set_xlabel('Épocas')
ax2.set_ylabel('F1-Score Macro')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/curvas_treinamento_mlp.png')
plt.show()

# Comparativo com Random Forest

print('\n--- Treinando Random Forest (Divisão Aleatória e Temporal) ---')

# 1. Random Forest no Dataset Aleatório
rf_aleatorio = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
rf_aleatorio.fit(X_train, y_train)
y_pred_rf_test = rf_aleatorio.predict(X_test)

# 2. Random Forest no Dataset Temporal
rf_temporal = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
rf_temporal.fit(X_train_temp, y_train_temp)
y_pred_rf_temp = rf_temporal.predict(X_test_temp)


# Função para extrair as 4 métricas multiclasse solicitadas
def obter_metricas_multi(y_true, y_pred):
  return {
      'Accuracy': accuracy_score(y_true, y_pred),
      'Precision Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'Recall Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'F1 Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
  }


# Montar tabela comparativa
df_comparacao_multi = pd.DataFrame([
    {
      'Modelo': 'MLP (Rede Neural)','Divisão': 'Aleatória',**obter_metricas_multi(y_test, y_pred_test),
    },
    {
        'Modelo': 'Random Forest','Divisão': 'Aleatória',**obter_metricas_multi(y_test, y_pred_rf_test),
    },
    {
        'Modelo': 'MLP (Rede Neural)','Divisão': 'Temporal',**obter_metricas_multi(y_test_temp, y_pred_temp),
    },
    {
        'Modelo': 'Random Forest','Divisão': 'Temporal',**obter_metricas_multi(y_test_temp, y_pred_rf_temp),
    },
])

print(df_comparacao_multi.to_string(index=False))