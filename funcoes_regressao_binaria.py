from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score
import numpy as np
from sklearn.metrics import roc_curve, auc


def superamostragem_classeminoritaria(dados, classe, valor_minoritario, valor_majoritario):
    # Suponha que sua base se chame df, com a coluna 'classe'
    majority = dados[dados[classe] == valor_majoritario]
    minority = dados[dados[classe] == valor_minoritario]

    # Superamostragem da classe minoritária
    minority_upsampled = resample(minority,
                                replace=True,    # com reposição
                                n_samples=len(majority),  # igual ao nº da majoritária
                                random_state=42)

    # Combinar as duas classes
    dados_superamostragem = pd.concat([majority, minority_upsampled])
    return dados_superamostragem

def subamostragem_classemajoritaria(dados, classe, valor_minoritario, valor_majoritario):
    # Suponha que sua base se chame df, com a coluna 'classe'
    majority = dados[dados[classe] == valor_majoritario]
    minority = dados[dados[classe] == valor_minoritario]

    # Subamostrando a classe majoritária
    majority_downsampled = resample(majority,
                                    replace=False,   # sem reposição
                                    n_samples=len(minority),  # igual ao nº da minoritária
                                    random_state=42)

    # Combinar as duas classes
    dados_subamostragem = pd.concat([majority_downsampled, minority])
    return dados_subamostragem

def boxplots(dados, excluir_classe):
    # Seleciona apenas variáveis numéricas (excluindo 'id', 'target' se necessário)
    dados_numericos = dados.select_dtypes(include='number').drop(columns=[excluir_classe], errors='ignore')

    # Define quantas colunas por linha você quer
    n_colunas = 4
    n_variaveis = len(dados_numericos.columns)
    n_linhas = (n_variaveis + n_colunas - 1) // n_colunas

    # Cria a figura com subplots
    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(5 * n_colunas, 4 * n_linhas))
    axes = axes.flatten()

    # Plota um boxplot por variável
    for i, coluna in enumerate(dados_numericos.columns):
        sns.boxplot(y=dados_numericos[coluna], ax=axes[i])
        axes[i].set_title(coluna)

    # Remove os eixos extras (se sobrarem)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def detectar_outliers_iqr(df, excluir_classe):
    # Seleciona colunas numéricas, excluindo a coluna de classe
    colunas_numericas = df.select_dtypes(include='number').drop(columns=excluir_classe, errors='ignore').columns

    resultado = {}

    for col in colunas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        AIQ = Q3 - Q1
        limite_inferior = Q1 - 1.5 * AIQ
        limite_superior = Q3 + 1.5 * AIQ
        
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)][col]

        resultado[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'AIQ': AIQ,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Limite Inferior': limite_inferior,
            'Limite Superior': limite_superior,
            'Qtd Outliers': outliers.count(),
        }

    return pd.DataFrame(resultado).T  # Transpõe para facilitar leitura

def substituir_outliers_mediana(df, excluir_classe):
    df_copy = df.copy()
    colunas_numericas = df_copy.select_dtypes(include='number').drop(columns = excluir_classe,  errors='ignore').columns
    
    for col in colunas_numericas:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        AIQ = Q3 - Q1
        limite_inferior = Q1 - 1.5 * AIQ
        limite_superior = Q3 + 1.5 * AIQ
        mediana = df_copy[col].median()
        
        # Substituir outliers pela mediana
        df_copy.loc[(df_copy[col] < limite_inferior) | (df_copy[col] > limite_superior), col] = mediana
        
    return df_copy

def heatmap(dados):
    # Mapa de calor das correlações entre as variáveis quantitativas
    plt.figure(figsize=(20,15))
    sns.heatmap(dados.corr(), annot=True, cmap = plt.cm.viridis,
                annot_kws={'size':25})
    plt.tight_layout()
    plt.show()

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualização dos principais indicadores da matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

def grafico_sensitividade_especificidade(observado, predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    #dita dos valores de sensitividade e especificidade em função dos cutoffs
    #(curva de sensibilidade)
    plt.figure(figsize=(7,5))
    plt.plot(resultado.cutoffs,resultado.sensitividade, '-o',
            color='darkorchid')
    plt.plot(resultado.cutoffs,resultado.especificidade, '-o',
            color='limegreen')
    plt.legend(['Sensitividade', 'Especificidade'], fontsize=17)
    plt.xlabel('Cuttoff', fontsize=17)
    plt.ylabel('Sensitividade / Especificidade', fontsize=17)
    plt.show()

def curva_roc(observado, previsto):
    # Função 'roc_curve' do pacote 'metrics' do sklearn
    fpr, tpr, thresholds = roc_curve(observado,previsto)
    roc_auc = auc(fpr, tpr)

    # Cálculo do coeficiente de GINI
    gini = (roc_auc - 0.5)/(0.5)

    # Plotagem da curva ROC propriamente dita
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, '-o', color='darkorchid')
    plt.plot(fpr, fpr, color='gray')
    plt.title('Área abaixo da curva: %g' % round(roc_auc,4) +
            ' | Coeficiente de GINI: %g' % round(gini,4), fontsize=17)
    plt.xlabel('1 - Especificidade', fontsize=17)
    plt.ylabel('Sensitividade', fontsize=17)
    plt.show()

