import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def possiveis_problemas(df):
    # Desbalanceamento de classes e classes com poucos exemplos
    print('DESBALANCEAMENTO E CLASSES COM POUCOS EXEMPLOS:')
    print('Frequência por classe:')
    print(df['classe'].value_counts())
    print(f"\nTotal de espécies distintas em 'especie_nm': {df['especie_nm'].nunique()}")

    especies_raras = (df['especie_nm'].value_counts() < 5).sum()
    print(f"Qtd. de espécies com menos de 5 registros: {especies_raras}\n")

    #Valores Ausentes e nao_identificado
    print('VALORES NULOS E "nao_identificado":')
    nao_identificado = (df.astype(str).apply(lambda col: col.str.lower().str.contains('nao_identificado')).sum())
    porcentagem_nao_identificado = (nao_identificado / len(df)) * 100
    df_nao_identificado = pd.DataFrame({'Qtd Nao_Identificado': nao_identificado, '%': porcentagem_nao_identificado.round(2)})
    print(df_nao_identificado[df_nao_identificado['Qtd Nao_Identificado'] > 0], '\n')

    # Outliers nas Coordenadas
    print('INVESTIGAÇÃO DE OUTLIERS NAS COORDENADAS:')

    # Calculo dos quartis
    primeiro_quartil_x = df['x'].quantile(0.25)
    terceiro_quartil_x = df['x'].quantile(0.75)
    primeiro_quartil_y = df['y'].quantile(0.25)
    terceiro_quartil_y = df['y'].quantile(0.75)

    # Calculo da Amplitude Interquartil
    amplitude_interquartil_x = terceiro_quartil_x - primeiro_quartil_x
    amplitude_interquartil_y = terceiro_quartil_y - primeiro_quartil_y

    # Definição dos limites de tolerância 
    limite_inferior_x = primeiro_quartil_x - (1.5 * amplitude_interquartil_x)
    limite_superior_x = terceiro_quartil_x + (1.5 * amplitude_interquartil_x)
    limite_inferior_y = primeiro_quartil_y - (1.5 * amplitude_interquartil_y)
    limite_superior_y = terceiro_quartil_y + (1.5 * amplitude_interquartil_y)

    # Filtragem dos registros que caem fora dos limites aceitáveis
    registros_outliers_x = df[(df['x'] < limite_inferior_x) | (df['x'] > limite_superior_x)]
    registros_outliers_y = df[(df['y'] < limite_inferior_y) | (df['y'] > limite_superior_y)]

    print(f'Total de potenciais outliers na coordenada X: {len(registros_outliers_x)}')
    print(f'Total de potenciais outliers na coordenada Y: {len(registros_outliers_y)}\n')
    print(f"Intervalo X: Mínimo = {df['x'].min()}, Máximo = {df['x'].max()}")
    print(f"Intervalo Y: Mínimo = {df['y'].min()}, Máximo = {df['y'].max()}\n")


def distribuicao_espacial_temporal(df):

    quantidade_uc = df['uc'].value_counts()
    porcentagem_uc = df['uc'].value_counts(normalize=True) * 100
    df_dist_uc = pd.DataFrame({'Ocorrências': quantidade_uc, 'Percentual (%)': porcentagem_uc.round(2)})
    print(df_dist_uc)
    print('\n')

    # Grafico Espacial
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='uc', alpha=0.6, palette='viridis', s=20)
    plt.title('Distribuição Espacial dos Atropelamentos', fontsize=12, fontweight='bold')
    plt.xlabel('X ', fontsize=10)
    plt.ylabel('Y ', fontsize=10)
    plt.legend(title='Unidade de Conservação (UC)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Tabela por Mês 
    df['data_dt'] = pd.to_datetime(df['data_dt'], format='%d/%m/%y', errors='coerce') 

    # Criar a coluna com o número do mês
    df['mes_num'] = df['data_dt'].dt.month
    distintos_mes = df['mes_num'].value_counts().sort_index()
    print(distintos_mes.to_frame(name='Ocorrências'))
    print('\n')

    # Tabela por Dia da Semana
    ordem_dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
    distintos_dias = df['dia_da_sem'].str.lower().value_counts().reindex(ordem_dias)
    print(distintos_dias.dropna().to_frame(name='Ocorrências'))

    #Gráficos Temporais 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico Mensal
    sns.barplot(x=distintos_mes.index,y=distintos_mes.values,ax=axes[0],hue=distintos_mes.index,palette='Blues_d',legend=False,)

    axes[0].set_title('Atropelamentos por Mês do Ano', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mês (1 = Jan, 12 = Dez)')
    axes[0].set_ylabel('Quantidade de Ocorrências')

    # Gráfico por Dia da Semana
    sns.barplot(x=distintos_dias.index,y=distintos_dias.values,ax=axes[1],hue=distintos_dias.index,palette='Oranges_d',legend=False,)
    axes[1].set_title('Atropelamentos por Dia da Semana', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dia da Semana')
    axes[1].set_ylabel('Quantidade de Ocorrências')

    plt.tight_layout()
    plt.show()

def analise_dados(df):
    
    
    print(f"Quantidade de Registros: {len(df)}")
    print(f"Quantidade de Atributos: {len(df.columns)}")

    df_quantidade_nulos = df.astype(str).apply(lambda col: col.str.lower().str.contains('nao_identificado')).sum()
    porcentagem_nulos = (df_quantidade_nulos / len(df)) * 100
    df_nao_id = pd.DataFrame({'Qtd Nao_Identificado': df_quantidade_nulos, 'Percentual (%)': porcentagem_nulos.round(2)})
    print(df_nao_id[df_nao_id['Qtd Nao_Identificado'] > 0])

    print(f"Tipos de Dados: {df.dtypes}")

    # Distribuição das Classes
    quantidade_classes = df['classe'].value_counts(dropna=False)
    porcentagem_classes = df['classe'].value_counts(normalize=True, dropna=False) * 100
    df_multiclasse = pd.DataFrame({'Quantidade': quantidade_classes, 'Percentual (%)': porcentagem_classes.round(2)})   
    print(df_multiclasse)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot( x = df_multiclasse.index, y = df_multiclasse['Quantidade'], palette='Blues_d', hue=0.5)
    plt.title('Distribuição dos Registros por Grupo Taxonômico (classe)',fontsize=12,fontweight='bold',)
    plt.xlabel('Grupo Taxonômico', fontsize=10)
    plt.ylabel('Quantidade de Atropelamentos', fontsize=10)
    for p in ax.patches:
        ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 5),
        textcoords='offset points',
        )

    plt.tight_layout()
    plt.show()

    # Análose Descritiva
    colunas_numericas = ['x', 'y', 'ano']
    estatisticas = df[colunas_numericas].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    estatisticas.columns = ['Contagem','Média','Desvio Padrão','Mínimo','Q1 (25%)','Mediana (50%)','Q3 (75%)','Máximo']
    print(estatisticas.round(2))

    distribuicao_espacial_temporal(df)
    possiveis_problemas(df)


caminho_arquivo = 'dataSet/ProgramaRodofaunaAtropelada.xlsx'
df = pd.read_excel(caminho_arquivo)
df = df[df["silvestre_"].astype(str).str.strip().str.lower() == "silvestre"].copy()
analise_dados(df)