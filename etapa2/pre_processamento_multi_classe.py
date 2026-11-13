import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


caminho_arquivo = 'dataSet/ProgramaRodofaunaAtropelada.xlsx'
df = pd.read_excel(caminho_arquivo)

# Filtrar apenas fauna silvestre e remover registros sem identificação taxonômica
df = df[df['silvestre_'].astype(str).str.strip().str.lower() == 'silvestre'].copy()
df = df[df['classe'].astype(str).str.lower() != 'nao_identificado'].copy()

# Limpeza e padronização das colunas de texto para evitar inconsistências
df['uc'] = df['uc'].astype(str).str.strip().str.lower()
df['dia_da_sem'] = df['dia_da_sem'].astype(str).str.strip().str.lower()

# Extração da Data e Tratamento de NaNs
df['data_dt'] = pd.to_datetime(df['data_dt'], format='%d/%m/%y', errors='coerce')
df = df.dropna(subset=['data_dt']).copy()
df['mes_num'] = df['data_dt'].dt.month

# Transformação Cíclica para a Sazonalidade Mensal (Seno e Cosseno)
df['mes_sin'] = np.sin(2 * np.pi * df['mes_num'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes_num'] / 12)

# Definição dos Preditores (X) e Alvo (y)
colunas_categoricas_nominais = ['uc', 'dia_da_sem']
colunas_numericas_continuas = ['x', 'y', 'ano', 'mes_sin', 'mes_cos']

X = df[colunas_categoricas_nominais + colunas_numericas_continuas].copy()

# Mapeamento numérico da variável alvo
le_target = LabelEncoder()
y = le_target.fit_transform(df['classe'])


# DIVISÃO ALEATÓRIA ESTRATIFICADA 
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val,
)

preprocessador_a = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), colunas_categoricas_nominais,),
        ('scaler', StandardScaler(), colunas_numericas_continuas),
    ]
)

X_train_proc = preprocessador_a.fit_transform(X_train)
X_val_proc = preprocessador_a.transform(X_val)
X_test_proc = preprocessador_a.transform(X_test)

# DIVISÃO TEMPORAL

mascara_treino_temp = df['ano'] <= 2013
mascara_teste_temp = df['ano'] >= 2014

X_train_temp = X[mascara_treino_temp]
y_train_temp = y[mascara_treino_temp]

X_test_temp = X[mascara_teste_temp]
y_test_temp = y[mascara_teste_temp]

preprocessador_b = ColumnTransformer(
    transformers=[
        ('onehot',OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),colunas_categoricas_nominais,),
        ('scaler', StandardScaler(), colunas_numericas_continuas),
    ]
)

X_train_temp_proc = preprocessador_b.fit_transform(X_train_temp)
X_test_temp_proc = preprocessador_b.transform(X_test_temp)


# EXPORTAÇÃO PARA CSV
os.makedirs('dataSet/treinoTesteValidacao', exist_ok=True)

colunas_proc_a = preprocessador_a.get_feature_names_out()
colunas_proc_b = preprocessador_b.get_feature_names_out()

# Estruturação dos DataFrames da Divisão Aleatória
df_train = pd.DataFrame(X_train_proc, columns=colunas_proc_a)
df_train['target'] = y_train

df_val = pd.DataFrame(X_val_proc, columns=colunas_proc_a)
df_val['target'] = y_val

df_test = pd.DataFrame(X_test_proc, columns=colunas_proc_a)
df_test['target'] = y_test

# Estruturação dos DataFrames da Divisão Temporal
df_train_temp = pd.DataFrame(X_train_temp_proc, columns=colunas_proc_b)
df_train_temp['target'] = y_train_temp

df_test_temp = pd.DataFrame(X_test_temp_proc, columns=colunas_proc_b)
df_test_temp['target'] = y_test_temp

# Escrita dos arquivos CSV na pasta 'dados_processados/'
df_train.to_csv('dataSet/treinoTesteValidacao/treino.csv', index=False)
df_val.to_csv('dataSet/treinoTesteValidacao/validacao.csv', index=False)
df_test.to_csv('dataSet/treinoTesteValidacao/teste.csv', index=False)

df_train_temp.to_csv('dataSet/treinoTesteValidacao/treino_temporal.csv', index=False)
df_test_temp.to_csv('dataSet/treinoTesteValidacao/teste_temporal.csv', index=False)