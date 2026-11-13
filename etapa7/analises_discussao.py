import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')
os.makedirs('img', exist_ok=True)
os.makedirs('resultados', exist_ok=True)
rng = np.random.default_rng(42)

print('--- CARREGANDO DADOS MULTICLASSE ---')
pasta = 'dataSet/treinoTesteValidacao/'
treino = pd.read_csv(pasta + 'treino.csv')
teste = pd.read_csv(pasta + 'teste.csv')
X_train, y_train = treino.drop(columns=['target']), treino['target'].values
X_test, y_test = teste.drop(columns=['target']), teste['target'].values
nomes_classes = ['Anfíbios', 'Aves', 'Mamíferos', 'Répteis']

# Configuração da MLP encontrada pelo Optuna (Etapa 5)
with open('resultados/etapa5_resumo.json', encoding='utf-8') as f:
  etapa5 = json.load(f)
p_otim = etapa5['params_otimizados']
epocas_otim = etapa5['melhor_epoca']


def treinar_mlp_otimizada(X, y, balancear=True):
  mlp = MLPClassifier(
      hidden_layer_sizes=tuple(p_otim['camadas']),
      activation=p_otim['activation'],
      solver=p_otim['solver'],
      alpha=p_otim['alpha'],
      learning_rate_init=p_otim['lr'],
      batch_size=p_otim['batch_size'],
      momentum=p_otim.get('momentum', 0.9),
      nesterovs_momentum=p_otim.get('nesterov', True),
      random_state=42,
  )
  pesos = compute_sample_weight('balanced', y) if balancear else None
  for _ in range(epocas_otim):
    mlp.partial_fit(X, y, classes=np.unique(y), sample_weight=pesos)
  return mlp


def criar_rf(balancear=True):
  # Mesma configuração da Etapa 3
  return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1,
                                class_weight='balanced' if balancear else None)


# 1. Importância dos atributos por permutação

print('\n--- 1. Importância dos atributos ---')
grupos = {
    'uc (Unidade de Conservação)': [c for c in X_test.columns if c.startswith('onehot__uc_')],
    'dia_da_sem': [c for c in X_test.columns if c.startswith('onehot__dia_da_sem_')],
    'x (UTM leste)': ['scaler__x'],
    'y (UTM norte)': ['scaler__y'],
    'ano': ['scaler__ano'],
    'mês (sen+cos)': ['scaler__mes_sin', 'scaler__mes_cos'],
}

mlp = treinar_mlp_otimizada(X_train, y_train)
rf = criar_rf().fit(X_train, y_train)


# Embaralha as colunas de cada atributo e mede quanto o F1 Macro cai
def importancia_permutacao(modelo, X, y):
  f1_base = f1_score(y, modelo.predict(X), average='macro')
  resultado = {}
  for nome, colunas in grupos.items():
    quedas = []
    for _ in range(10):
      X_emb = X.copy()
      idx = rng.permutation(len(X_emb))
      X_emb[colunas] = X_emb[colunas].values[idx]
      quedas.append(f1_base - f1_score(y, modelo.predict(X_emb), average='macro'))
    resultado[nome] = np.mean(quedas)
  return resultado


df_imp = pd.DataFrame({
    'MLP otimizada': importancia_permutacao(mlp, X_test, y_test),
    'Random Forest': importancia_permutacao(rf, X_test, y_test),
}).sort_values('Random Forest')
print(df_imp.round(4).to_string())
df_imp.to_csv('resultados/etapa7_importancia_permutacao.csv')

# Gráfico 1: importância dos atributos
ax = df_imp.plot(kind='barh', figsize=(9, 5), color=['tab:blue', 'tab:green'])
ax.set_xlabel('Queda do F1-Macro no teste ao embaralhar o atributo')
ax.set_title('Importância dos atributos (permutação) – Multiclasse', fontsize=12, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('img/etapa7_importancia_atributos.png', dpi=120)
plt.show()


# 2. Impacto do desbalanceamento: com e sem pesos de classe

print('\n--- 2. Impacto do desbalanceamento ---')
linhas = []
for balancear in (False, True):
  modelos = {
      'MLP otimizada': treinar_mlp_otimizada(X_train, y_train, balancear),
      'Random Forest': criar_rf(balancear).fit(X_train, y_train),
  }
  for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    recall = recall_score(y_test, y_pred, average=None)
    linha = {'Modelo': nome, 'Pesos de classe': 'sim' if balancear else 'não',
             'Accuracy': accuracy_score(y_test, y_pred),
             'F1 Macro': f1_score(y_test, y_pred, average='macro')}
    for c, r in zip(nomes_classes, recall):
      linha[f'Recall {c}'] = r
    linhas.append(linha)

df_bal = pd.DataFrame(linhas).sort_values(['Modelo', 'Pesos de classe'])
print(df_bal.round(4).to_string(index=False))
df_bal.to_csv('resultados/etapa7_desbalanceamento.csv', index=False)

# Linha de base: um modelo que sempre responde "Aves"
print(f'Sempre "Aves": accuracy={(y_test == 1).mean():.4f}, '
      f'F1 Macro={f1_score(y_test, np.ones_like(y_test), average="macro"):.4f}')


# 3. Complexidade da rede x desempenho (trials do Optuna)

print('\n--- 3. Complexidade x desempenho ---')
trials = pd.read_csv('resultados/optuna_trials.csv')
trials = trials[trials['state'] == 'COMPLETE'].copy()
colunas_neuronios = [c for c in trials.columns if c.startswith('params_neuronios_c')]
trials['total_neuronios'] = trials[colunas_neuronios].fillna(0).sum(axis=1)

df_comp = trials.groupby('params_n_camadas')['value'].agg(['count', 'mean', 'max']).round(4)
print(df_comp.to_string())
df_comp.to_csv('resultados/etapa7_complexidade_optuna.csv')

# Gráfico 2: complexidade x F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
trials.boxplot(column='value', by='params_n_camadas', ax=ax1,
               flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
ax1.set_title('F1 de validação por nº de camadas ocultas')
ax1.set_xlabel('Camadas ocultas\n(círculos = trials discrepantes do grupo)')
ax1.set_ylabel('F1-Macro (validação)')

cores = {1: 'tab:purple', 2: 'tab:blue', 3: 'tab:green', 4: 'tab:orange'}
for n, grupo in trials.groupby('params_n_camadas'):
  rotulo = '1 camada' if n == 1 else f'{int(n)} camadas'
  ax2.scatter(grupo['total_neuronios'], grupo['value'], color=cores[int(n)], s=40,
              edgecolor='black', linewidth=0.4, label=rotulo)
ax2.set_xscale('log')
ax2.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400, 800]))
ax2.xaxis.set_minor_locator(NullLocator())
ax2.set_xticklabels(['50', '100', '200', '400', '800'])
ax2.set_title('F1 de validação pelo total de neurônios')
ax2.set_xlabel('Total de neurônios da rede (escala log)\n(cada ponto = um trial completo do Optuna)')
ax2.legend(title='Camadas ocultas', fontsize=8, title_fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)
plt.suptitle('Complexidade da MLP × desempenho (trials do Optuna)', fontweight='bold')
plt.tight_layout()
plt.savefig('img/etapa7_complexidade.png', dpi=120)
plt.show()


# 4. Generalização espacial: treina em 2 UCs e testa na 3ª

print('\n--- 4. Generalização espacial (UC não vista) ---')
df = pd.read_excel('dataSet/ProgramaRodofaunaAtropelada.xlsx')
df = df[df['silvestre_'].astype(str).str.strip().str.lower() == 'silvestre']
df = df[df['classe'].astype(str).str.lower() != 'nao_identificado'].copy()
df['uc'] = df['uc'].astype(str).str.strip().str.lower()
df['dia_da_sem'] = df['dia_da_sem'].astype(str).str.strip().str.lower().replace({'eg_': 'seg'})
df['data_dt'] = pd.to_datetime(df['data_dt'], format='%d/%m/%y', errors='coerce')
df = df.dropna(subset=['data_dt'])
df['mes_sin'] = np.sin(2 * np.pi * df['data_dt'].dt.month / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['data_dt'].dt.month / 12)
df = df[df['uc'] != 'nao_identificado']

y_todos = LabelEncoder().fit_transform(df['classe'])
numericas = ['x', 'y', 'ano', 'mes_sin', 'mes_cos']
categoricas = ['dia_da_sem']  # a uc fica de fora, pois a UC de teste nunca foi vista

linhas = []
for uc_teste in sorted(df['uc'].unique()):
  mask_treino = (df['uc'] != uc_teste).values
  mask_teste = (df['uc'] == uc_teste).values

  prep = ColumnTransformer([
      ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categoricas),
      ('scaler', StandardScaler(), numericas),
  ])
  Xtr = pd.DataFrame(prep.fit_transform(df.loc[mask_treino, categoricas + numericas]))
  Xte = pd.DataFrame(prep.transform(df.loc[mask_teste, categoricas + numericas]))

  modelos = {
      'MLP otimizada': treinar_mlp_otimizada(Xtr, y_todos[mask_treino]),
      'Random Forest': criar_rf().fit(Xtr, y_todos[mask_treino]),
  }
  for nome, modelo in modelos.items():
    y_pred = modelo.predict(Xte)
    linhas.append({'UC de teste (não vista)': uc_teste, 'Modelo': nome, 'N teste': int(mask_teste.sum()),
                   'Accuracy': accuracy_score(y_todos[mask_teste], y_pred),
                   'F1 Macro': f1_score(y_todos[mask_teste], y_pred, average='macro')})

df_espacial = pd.DataFrame(linhas)
print(df_espacial.round(4).to_string(index=False))
print(df_espacial.groupby('Modelo')[['Accuracy', 'F1 Macro']].mean().round(4).to_string())
df_espacial.to_csv('resultados/etapa7_generalizacao_espacial_multiclasse.csv', index=False)


# 5. Hotspots (segmentos de 500 m): validação cruzada aleatória x por UC

print('\n--- 5. Hotspots: CV aleatória x CV por UC ---')
df['segmento_x'] = (df['x'] // 500) * 500
df['segmento_y'] = (df['y'] // 500) * 500
seg = df.groupby(['segmento_x', 'segmento_y']).agg(
    quantidade_atropelamentos=('id', 'count'),
    uc=('uc', lambda s: s.mode()[0]),
    x_centro=('x', 'mean'),
    y_centro=('y', 'mean'),
).reset_index()

limiar = seg['quantidade_atropelamentos'].quantile(0.75)
seg['alto_risco'] = (seg['quantidade_atropelamentos'] >= limiar).astype(int)
X_seg = seg[['x_centro', 'y_centro']]
y_seg = seg['alto_risco'].values
print(f'Segmentos: {len(seg)} | alto risco: >= {limiar:.0f} atropelamentos')

modelos_seg = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced',
                                                    random_state=42, n_jobs=-1),
    'MLP': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000,
                                 learning_rate_init=0.003, random_state=42),
}
linhas = []
prob_rf = None
for nome, criar in modelos_seg.items():
  cv_aleatoria = StratifiedKFold(5, shuffle=True, random_state=42)
  cv_por_uc = GroupKFold(n_splits=seg['uc'].nunique())

  prob_aleatoria = cross_val_predict(Pipeline([('scaler', StandardScaler()), ('modelo', criar())]),
                                     X_seg, y_seg, method='predict_proba', cv=cv_aleatoria)[:, 1]
  prob_uc = cross_val_predict(Pipeline([('scaler', StandardScaler()), ('modelo', criar())]),
                              X_seg, y_seg, method='predict_proba', cv=cv_por_uc, groups=seg['uc'])[:, 1]
  if nome == 'Random Forest':
    prob_rf = prob_aleatoria

  linhas.append({'Modelo': nome,
                 'ROC-AUC CV aleatória (5 folds)': roc_auc_score(y_seg, prob_aleatoria),
                 'ROC-AUC CV por UC (UC não vista)': roc_auc_score(y_seg, prob_uc)})

df_hotspots = pd.DataFrame(linhas)
print(df_hotspots.round(4).to_string(index=False))
df_hotspots.to_csv('resultados/etapa7_hotspots_cv.csv', index=False)

seg['prob_rf_oof'] = prob_rf
seg.to_csv('resultados/etapa7_segmentos_probabilidade.csv', index=False)

# Gráfico 3: mapa de hotspots (observado x previsto)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
sc = ax1.scatter(seg['x_centro'], seg['y_centro'], c=seg['quantidade_atropelamentos'], cmap='YlOrRd',
                 s=25, edgecolor='k', linewidth=0.2)
plt.colorbar(sc, ax=ax1, label='Atropelamentos observados (2010–2015)')
ax1.set_title('Observado: atropelamentos por célula de 500 m')

sc = ax2.scatter(seg['x_centro'], seg['y_centro'], c=seg['prob_rf_oof'], cmap='RdYlGn_r', vmin=0, vmax=1,
                 s=25, edgecolor='k', linewidth=0.2)
alto = seg[seg['alto_risco'] == 1]
ax2.scatter(alto['x_centro'], alto['y_centro'], s=90, facecolor='none', edgecolor='black', linewidth=1,
            label='Hotspot observado (top 25%)')
plt.colorbar(sc, ax=ax2, label='Probabilidade prevista de alto risco')
ax2.set_title('Previsto (Random Forest, fora da amostra)')
ax2.legend(loc='lower left')

for ax in (ax1, ax2):
  ax.set_xlabel('UTM X (m)')
  ax.grid(True, alpha=0.3)
ax1.set_ylabel('UTM Y (m)')
plt.suptitle('Mapa de hotspots de atropelamento de fauna silvestre – DF', fontweight='bold')
plt.tight_layout()
plt.savefig('img/etapa7_mapa_hotspots.png', dpi=120)
plt.show()
