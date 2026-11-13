import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


caminho_arquivo = 'dataSet/ProgramaRodofaunaAtropelada.xlsx'
df = pd.read_excel(caminho_arquivo)

df = df[df['silvestre_'].astype(str).str.strip().str.lower() == 'silvestre'].copy()
df['uc'] = df['uc'].astype(str).str.strip().str.lower()

# 2. Criar Segmentos Espaciais de 500 metros
df['segmento_x'] = (df['x'] // 500) * 500
df['segmento_y'] = (df['y'] // 500) * 500

# 3. Agregação por Segmento (Contagem de Atropelamentos e Predatores Espaciais)
df_segmentos = (
    df.groupby(['segmento_x', 'segmento_y']).agg(
        quantidade_atropelamentos=('id', 'count'), 
        uc=('uc', lambda x: x.mode()[0] if not x.empty else 'nao_identificado'),
        x_centro=('x', 'mean'),
        y_centro=('y', 'mean'),
    ).reset_index()
)

# 4. Criar a Variável Alvo Binária
limiar_75 = df_segmentos['quantidade_atropelamentos'].quantile(0.75)
df_segmentos['alto_risco'] = (df_segmentos['quantidade_atropelamentos'] >= limiar_75).astype(int)

print(f'Total de Segmentos de 500m identificados: {len(df_segmentos)}')
print(f'Limiar do Top 25% (Alto Risco): >= {limiar_75:.0f} atropelamentos por'' segmento')
print(f"Distribuição Alto Risco (1): {df_segmentos['alto_risco'].sum()} | Baixo"f" Risco (0): {(df_segmentos['alto_risco'] == 0).sum()}")

# 5. Definição de Preditores e Divisão para os Segmentos
colunas_categoricas = ['uc']
colunas_numericas = ['x_centro', 'y_centro']

X_seg = df_segmentos[colunas_categoricas + colunas_numericas]
y_reg = df_segmentos['quantidade_atropelamentos']
y_bin = df_segmentos['alto_risco']

# Divisão Treino / Validação / Teste (Estratificada pelo alto_risco)
X_train_val, X_test, y_bin_tr_val, y_bin_test, y_reg_tr_val, y_reg_test = (
    train_test_split(
        X_seg,
        y_bin,
        y_reg,
        test_size=0.15,
        random_state=42,
        stratify=y_bin,
    )
)

X_train, X_val, y_bin_train, y_bin_val, y_reg_train, y_reg_val = (
    train_test_split(
        X_train_val,
        y_bin_tr_val,
        y_reg_tr_val,
        test_size=0.1765,
        random_state=42,
        stratify=y_bin_tr_val,
    )
)

# Preprocessador para o dataset segmentado
preprocessador_seg = ColumnTransformer(
    transformers=[
        ('onehot',OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),colunas_categoricas,),
        ('scaler', StandardScaler(), colunas_numericas),
    ]
)

X_train_proc = preprocessador_seg.fit_transform(X_train)
X_val_proc = preprocessador_seg.transform(X_val)
X_test_proc = preprocessador_seg.transform(X_test)

colunas_proc = preprocessador_seg.get_feature_names_out()

# Exportar CSVs dos Segmentos para 'dataSet/treinoTesteValidacaoSegmentos'
os.makedirs('dataSet/treinoTesteValidacaoSegmentos', exist_ok=True)

df_tr = pd.DataFrame(X_train_proc, columns=colunas_proc)
df_tr['target_regressao'] = y_reg_train.values
df_tr['target_binario'] = y_bin_train.values

df_v = pd.DataFrame(X_val_proc, columns=colunas_proc)
df_v['target_regressao'] = y_reg_val.values
df_v['target_binario'] = y_bin_val.values

df_te = pd.DataFrame(X_test_proc, columns=colunas_proc)
df_te['target_regressao'] = y_reg_test.values
df_te['target_binario'] = y_bin_test.values

df_tr.to_csv('dataSet/treinoTesteValidacaoSegmentos/treino_segmentos.csv', index=False)
df_v.to_csv('dataSet/treinoTesteValidacaoSegmentos/validacao_segmentos.csv', index=False)
df_te.to_csv('dataSet/treinoTesteValidacaoSegmentos/teste_segmentos.csv', index=False)

print('\nDatasets por segmento salvos com sucesso!')