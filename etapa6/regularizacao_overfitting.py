import copy
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')
os.makedirs('img', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

print('--- CARREGANDO DADOS MULTICLASSE ---')
pasta = 'dataSet/treinoTesteValidacao/'
treino = pd.read_csv(pasta + 'treino.csv')
validacao = pd.read_csv(pasta + 'validacao.csv')
teste = pd.read_csv(pasta + 'teste.csv')
treino_temp = pd.read_csv(pasta + 'treino_temporal.csv')
teste_temp = pd.read_csv(pasta + 'teste_temporal.csv')

X_train, y_train = treino.drop(columns=['target']), treino['target'].values
X_val, y_val = validacao.drop(columns=['target']), validacao['target'].values
X_test, y_test = teste.drop(columns=['target']), teste['target'].values
X_train_temp, y_train_temp = treino_temp.drop(columns=['target']), treino_temp['target'].values
X_test_temp, y_test_temp = teste_temp.drop(columns=['target']), teste_temp['target'].values

classes = np.unique(y_train)


def criar_mlp(p):
  return MLPClassifier(
      hidden_layer_sizes=tuple(p['camadas']),
      activation=p['activation'],
      solver=p['solver'],
      alpha=p['alpha'],
      learning_rate_init=p['lr'],
      batch_size=p['batch_size'],
      momentum=p.get('momentum', 0.9),
      nesterovs_momentum=p.get('nesterov', True),
      random_state=42,
  )


# Loss com pesos balanceados (a mesma que a rede minimiza no treino)
def loss_ponderada(mlp, X, y):
  return log_loss(y, mlp.predict_proba(X), labels=classes, sample_weight=compute_sample_weight('balanced', y))


# Configurações comparadas
rede_grande = [256, 256, 128]
configs = [
    {'nome': 'A) Sem regularização', 'paciencia': None,
     'params': {'camadas': rede_grande, 'activation': 'relu', 'solver': 'adam', 'lr': 0.001, 'batch_size': 64, 'alpha': 0.0}},
    {'nome': 'B) L2 / weight decay (alpha=0.1)', 'paciencia': None,
     'params': {'camadas': rede_grande, 'activation': 'relu', 'solver': 'adam', 'lr': 0.001, 'batch_size': 64, 'alpha': 0.1}},
    {'nome': 'C) Early stopping (paciência=20)', 'paciencia': 20,
     'params': {'camadas': rede_grande, 'activation': 'relu', 'solver': 'adam', 'lr': 0.001, 'batch_size': 64, 'alpha': 0.0}},
    {'nome': 'D) Rede reduzida (32, 16)', 'paciencia': None,
     'params': {'camadas': [32, 16], 'activation': 'relu', 'solver': 'adam', 'lr': 0.001, 'batch_size': 64, 'alpha': 0.0}},
]

# Configuração encontrada pelo Optuna na Etapa 5
with open('resultados/etapa5_resumo.json', encoding='utf-8') as f:
  p_optuna = json.load(f)['params_otimizados']
configs.append({'nome': 'E) Config. Optuna + early stopping', 'paciencia': 20, 'params': p_optuna})

max_epocas = 300
resultados = []
historicos = {}

for cfg in configs:
  print('\n---', cfg['nome'], '---')
  mlp = criar_mlp(cfg['params'])
  pesos = compute_sample_weight('balanced', y_train)

  hist = {'loss_treino': [], 'loss_val': [], 'f1_treino': [], 'f1_val': []}
  melhor_f1, melhor_epoca, sem_melhora, melhor_mlp = -1, 0, 0, None
  inicio = time.time()

  for epoca in range(1, max_epocas + 1):
    mlp.partial_fit(X_train, y_train, classes=classes, sample_weight=pesos)

    f1_val = f1_score(y_val, mlp.predict(X_val), average='macro')
    hist['loss_treino'].append(loss_ponderada(mlp, X_train, y_train))
    hist['loss_val'].append(loss_ponderada(mlp, X_val, y_val))
    hist['f1_treino'].append(f1_score(y_train, mlp.predict(X_train), average='macro'))
    hist['f1_val'].append(f1_val)

    if f1_val > melhor_f1:
      melhor_f1, melhor_epoca, sem_melhora = f1_val, epoca, 0
      melhor_mlp = copy.deepcopy(mlp)
    else:
      sem_melhora += 1

    # Early stopping: para após 'paciencia' épocas sem melhora
    if cfg['paciencia'] is not None and sem_melhora >= cfg['paciencia']:
      break

  tempo = time.time() - inicio

  # Com early stopping, volta para os pesos da melhor época
  if cfg['paciencia'] is not None:
    mlp = melhor_mlp
    epocas_usadas = melhor_epoca
  else:
    epocas_usadas = epoca

  historicos[cfg['nome']] = (hist, melhor_epoca)

  f1_tr = f1_score(y_train, mlp.predict(X_train), average='macro')
  f1_va = f1_score(y_val, mlp.predict(X_val), average='macro')
  y_pred = mlp.predict(X_test)
  f1_te = f1_score(y_test, y_pred, average='macro')

  # Teste temporal: treina de novo em 2010-2013 com o mesmo número de épocas
  mlp_temp = criar_mlp(cfg['params'])
  pesos_temp = compute_sample_weight('balanced', y_train_temp)
  for _ in range(epocas_usadas):
    mlp_temp.partial_fit(X_train_temp, y_train_temp, classes=classes, sample_weight=pesos_temp)
  f1_temp = f1_score(y_test_temp, mlp_temp.predict(X_test_temp), average='macro')

  resultados.append({
      'Configuração': cfg['nome'],
      'Épocas usadas': epocas_usadas,
      'F1 treino': f1_tr,
      'F1 validação': f1_va,
      'Gap treino−val': f1_tr - f1_va,
      'Loss treino (final)': hist['loss_treino'][epocas_usadas - 1],
      'Loss val (final)': hist['loss_val'][epocas_usadas - 1],
      'Loss val (mínima)': min(hist['loss_val']),
      'Acc teste': accuracy_score(y_test, y_pred),
      'F1 teste (aleatório)': f1_te,
      'F1 teste temporal (2014–15)': f1_temp,
      'Tempo (s)': tempo,
  })
  print(f'épocas={epocas_usadas} | F1 treino={f1_tr:.4f} | F1 val={f1_va:.4f} | F1 teste={f1_te:.4f} | F1 temporal={f1_temp:.4f}')

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('resultados/etapa6_regularizacao.csv', index=False)
print('\n--- MLP SEM x COM REGULARIZAÇÃO ---')
print(df_resultados.round(4).to_string(index=False))


# Gráfico 1: curvas de treino e validação de cada configuração
n = len(historicos)
fig, eixos = plt.subplots(2, n, figsize=(4.2 * n, 7.5), sharey='row')

for j, (nome, (hist, melhor_epoca)) in enumerate(historicos.items()):
  epocas = range(1, len(hist['f1_val']) + 1)
  ax_loss, ax_f1 = eixos[0, j], eixos[1, j]

  ax_loss.plot(epocas, hist['loss_treino'], color='blue', label='Treino')
  ax_loss.plot(epocas, hist['loss_val'], color='orange', linestyle='--', label='Validação')
  ax_f1.plot(epocas, hist['f1_treino'], color='green', label='Treino')
  ax_f1.plot(epocas, hist['f1_val'], color='red', linestyle='--', label='Validação')

  for ax in (ax_loss, ax_f1):
    if 'Early' in nome or 'Optuna' in nome:
      ax.axvline(melhor_epoca, color='gray', linestyle=':', label='Melhor época')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Épocas')

  ax_loss.set_title(nome, fontsize=10, fontweight='bold')
  if j == 0:
    ax_loss.set_ylabel('Loss (log-loss ponderada)')
    ax_f1.set_ylabel('F1-Macro')
    ax_loss.legend(fontsize=8)
    ax_f1.legend(fontsize=8)

fig.suptitle('Etapa 6 – Curvas de Treino e Validação: MLP sem × com Regularização', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('img/etapa6_curvas_regularizacao.png', dpi=110)
plt.show()

# Gráfico 2: F1 em cada conjunto
fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(df_resultados))
larg = 0.2
ax.bar(pos - 1.5 * larg, df_resultados['F1 treino'], larg, label='Treino')
ax.bar(pos - 0.5 * larg, df_resultados['F1 validação'], larg, label='Validação')
ax.bar(pos + 0.5 * larg, df_resultados['F1 teste (aleatório)'], larg, label='Teste aleatório')
ax.bar(pos + 1.5 * larg, df_resultados['F1 teste temporal (2014–15)'], larg, label='Teste temporal')
ax.set_xticks(pos)
ax.set_xticklabels([c.split(')')[0] + ')' for c in df_resultados['Configuração']])
ax.set_ylabel('F1-Macro')
ax.set_title('F1-Macro por conjunto – efeito da regularização', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('img/etapa6_gap_generalizacao.png', dpi=120)
plt.show()
