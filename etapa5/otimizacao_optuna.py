import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
nomes_classes = ['Anfíbios', 'Aves', 'Mamíferos', 'Répteis']


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


# Função objetivo: maximizar o F1 Macro na validação (o teste não é usado aqui)
def objetivo(trial):
  n_camadas = trial.suggest_int('n_camadas', 1, 4)
  camadas = [trial.suggest_int(f'neuronios_c{i + 1}', 16, 256, log=True) for i in range(n_camadas)]
  p = {
      'camadas': camadas,
      'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
      'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
      'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
      'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
      'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
  }
  max_epocas = trial.suggest_int('max_epocas', 50, 300, step=50)
  if p['solver'] == 'sgd':
    p['momentum'] = trial.suggest_float('momentum', 0.5, 0.99)
    p['nesterov'] = trial.suggest_categorical('nesterov', [True, False])

  mlp = criar_mlp(p)
  pesos = compute_sample_weight('balanced', y_train)
  melhor_f1, melhor_epoca, sem_melhora = -1, 0, 0

  for epoca in range(1, max_epocas + 1):
    mlp.partial_fit(X_train, y_train, classes=classes, sample_weight=pesos)
    f1_val = f1_score(y_val, mlp.predict(X_val), average='macro')

    if f1_val > melhor_f1:
      melhor_f1, melhor_epoca, sem_melhora = f1_val, epoca, 0
    else:
      sem_melhora += 1

    # Pruning: o Optuna interrompe trials piores que a mediana dos anteriores
    trial.report(f1_val, epoca)
    if trial.should_prune():
      raise optuna.TrialPruned()

    # Early stopping: para após 20 épocas sem melhora
    if sem_melhora >= 20:
      break

  trial.set_user_attr('melhor_epoca', melhor_epoca)
  return melhor_f1


print('\n--- Rodando o Optuna (80 trials) ---')
optuna.logging.set_verbosity(optuna.logging.WARNING)
estudo = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
)
inicio = time.time()
estudo.optimize(objetivo, n_trials=80)
tempo_busca = time.time() - inicio

df_trials = estudo.trials_dataframe()
df_trials.to_csv('resultados/optuna_trials.csv', index=False)

print('Trials:', df_trials['state'].value_counts().to_dict())
print(f'Tempo da busca: {tempo_busca / 60:.1f} min')
print(f'Melhor trial: {estudo.best_trial.number}')
print(f'Melhor F1 Macro (validação): {estudo.best_value:.4f}')
print('Melhores hiperparâmetros:', estudo.best_params)

# Gráfico 1: histórico da otimização
completos = df_trials[df_trials['state'] == 'COMPLETE']
podados = df_trials[df_trials['state'] == 'PRUNED']

plt.figure(figsize=(9, 5))
plt.scatter(completos['number'], completos['value'], s=18, color='steelblue', label='F1-Macro do trial (validação)')
plt.plot(completos['number'], completos['value'].cummax(), color='red', linewidth=2, label='Melhor valor até o trial')
plt.scatter(podados['number'], [completos['value'].min()] * len(podados), marker='x', color='gray', s=18, label='Trials interrompidos (pruning)')
plt.title('Histórico da Otimização (Optuna)', fontsize=12, fontweight='bold')
plt.xlabel('Trial')
plt.ylabel('F1-Macro na validação')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/optuna_historico.png', dpi=120)
plt.show()

# Gráfico 2: importância dos hiperparâmetros
importancias = optuna.importance.get_param_importances(
    estudo, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
)
imp = pd.Series(importancias).sort_values()
print('\nImportância dos hiperparâmetros:')
print(imp.sort_values(ascending=False).round(3).to_string())

plt.figure(figsize=(8, 5))
imp.plot(kind='barh', color='teal')
plt.title('Importância dos Hiperparâmetros (fANOVA)', fontsize=12, fontweight='bold')
plt.xlabel('Importância relativa')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('img/optuna_importancia.png', dpi=120)
plt.show()


# Comparação: MLP original (Etapa 3) x MLP otimizada

bp = estudo.best_params
p_otim = {
    'camadas': [bp[f'neuronios_c{i + 1}'] for i in range(bp['n_camadas'])],
    'activation': bp['activation'],
    'solver': bp['solver'],
    'lr': bp['lr'],
    'batch_size': bp['batch_size'],
    'alpha': bp['alpha'],
    'momentum': bp.get('momentum', 0.9),
    'nesterov': bp.get('nesterov', True),
}
epocas_otim = estudo.best_trial.user_attrs['melhor_epoca']
print('\nConfiguração otimizada:', p_otim, '| épocas:', epocas_otim)


def calcular_metricas(y_true, y_pred):
  return {
      'Accuracy': accuracy_score(y_true, y_pred),
      'Precision Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'Recall Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'F1 Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
  }


divisoes = {
    'Aleatória': (X_train, y_train, X_test, y_test),
    'Temporal (2014–2015)': (X_train_temp, y_train_temp, X_test_temp, y_test_temp),
}
linhas = []

for nome, (Xtr, ytr, Xte, yte) in divisoes.items():
  pesos = compute_sample_weight('balanced', ytr)

  # MLP original, mesma configuração da Etapa 3
  inicio = time.time()
  mlp_orig = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                           alpha=0.05, learning_rate_init=0.0005, max_iter=400,
                           random_state=42, batch_size=200)
  mlp_orig.fit(Xtr, ytr, sample_weight=pesos)
  tempo_orig = time.time() - inicio
  linhas.append({'Modelo': 'MLP original', 'Divisão': nome,
                 **calcular_metricas(yte, mlp_orig.predict(Xte)),
                 'Tempo treino (s)': tempo_orig, 'Épocas': mlp_orig.n_iter_})

  # MLP otimizada, treinada pelo número de épocas escolhido na validação
  inicio = time.time()
  mlp_otim = criar_mlp(p_otim)
  for _ in range(epocas_otim):
    mlp_otim.partial_fit(Xtr, ytr, classes=classes, sample_weight=pesos)
  tempo_otim = time.time() - inicio
  y_pred_otim = mlp_otim.predict(Xte)
  linhas.append({'Modelo': 'MLP otimizada', 'Divisão': nome,
                 **calcular_metricas(yte, y_pred_otim),
                 'Tempo treino (s)': tempo_otim, 'Épocas': epocas_otim})

  if nome == 'Aleatória':
    print('\nDesempenho por classe - MLP otimizada (teste aleatório):')
    print(classification_report(yte, y_pred_otim, target_names=nomes_classes, digits=4))

    # Gráfico 3: matriz de confusão da MLP otimizada
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(yte, y_pred_otim), annot=True, fmt='d', cmap='Greens',
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title('Matriz de Confusão – MLP Otimizada (Teste Aleatório)', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    plt.savefig('img/matriz_confusao_otimizada.png', dpi=120)
    plt.show()

df_comparacao = pd.DataFrame(linhas)
df_comparacao.to_csv('resultados/etapa5_original_vs_otimizado.csv', index=False)
print('\n--- MLP ORIGINAL x MLP OTIMIZADA (TESTE) ---')
print(df_comparacao.round(4).to_string(index=False))

# Salva a configuração otimizada para as Etapas 6 e 7
with open('resultados/etapa5_resumo.json', 'w', encoding='utf-8') as f:
  json.dump({
      'melhor_trial': estudo.best_trial.number,
      'melhor_f1_val': estudo.best_value,
      'melhores_params': estudo.best_params,
      'melhor_epoca': epocas_otim,
      'params_otimizados': p_otim,
      'importancias': importancias,
      'tempo_busca_min': tempo_busca / 60,
  }, f, ensure_ascii=False, indent=2, default=float)
