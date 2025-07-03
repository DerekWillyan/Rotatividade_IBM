# Análise - Rotatividade de Funcionários da IBM

Este repositório contém um projeto completo de análise e modelagem preditiva para estudar a rotatividade de funcionários da IBM, utilizando técnicas de Análise Exploratória de Dados (EDA), Regressão Logística e Machine Learning com XGBoost.

---

## 📁 Estrutura do Projeto

- **analysis.ipynb**  
  Notebook com a análise exploratória detalhada dos dados e desenvolvimento do modelo de Regressão Logística Binária para previsão da rotatividade.

- **xgboostModel.ipynb**  
  Notebook com o desenvolvimento do modelo preditivo utilizando o algoritmo XGBoost, incluindo ajuste de hiperparâmetros e interpretação com SHAP.

- **funcoes_regressao_binaria.py**  
  Módulo Python contendo funções auxiliares para avaliação e visualização de modelos de regressão logística e classificação binária, usadas ao longo dos notebooks.

- **dados_ibm_rotatividade.csv**  
  Dataset final, tratado e balanceado, utilizado para treinamento e validação dos modelos.

- **hr.csv**  
  Dados brutos com informações pessoais, funcionais e demográficas dos funcionários.

- **years.csv**  
  Dados complementares relacionados ao tempo de serviço e histórico na empresa, que foram pivotados e integrados ao dataset principal.

---

## 🚀 Objetivos do Projeto

- Analisar os principais fatores relacionados à rotatividade dos funcionários da IBM.
- Construir modelos preditivos para identificar quais colaboradores têm maior probabilidade de saída.
- Comparar abordagens tradicionais (Regressão Logística) e métodos avançados de machine learning (XGBoost).
- Utilizar técnicas de interpretação de modelo para explicar os principais drivers da rotatividade.
- Fornecer insights acionáveis para reduzir a rotatividade e melhorar a gestão de pessoas.

---

## 🔧 Tecnologias e Bibliotecas Utilizadas

- Python 3.x  
- Pandas, NumPy, Matplotlib, Seaborn (Análise e visualização de dados)  
- statsmodels (Modelagem estatística e regressão logística)  
- scikit-learn (Pré-processamento, avaliação e validação)  
- XGBoost (Modelagem avançada de classificação)  
- SHAP (Interpretação dos modelos de machine learning)  
- Imbalanced-learn (Técnicas de balanceamento de dados, como SMOTE)  

---

## 📋 Como Executar

1. Clone este repositório:  
   ```bash
   git clone https://github.com/DerekWillyan/Rotatividade_IBM.git
   cd Analise-Rotatividade-IBM
   ```

2. Instale as dependências (recomendado usar ambiente virtual):

   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks na ordem recomendada:

   * `analysis.ipynb` para análise exploratória e regressão logística.
   * `xgboostModel.ipynb` para o modelo XGBoost e interpretação.

4. Explore os resultados, gráficos e relatórios gerados nos notebooks.

---

## 📊 Resultados Principais

* Modelos precisos para previsão da rotatividade com acurácia superior a 90%.
* Identificação das variáveis mais influentes que impactam a saída dos funcionários.
* Visualizações detalhadas que auxiliam na tomada de decisão estratégica.

---

## 📞 Contato

Para dúvidas, sugestões ou contribuições, sinta-se à vontade para abrir uma issue ou enviar um pull request.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT - consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

*Desenvolvido por Derek Willyan*
*Última atualização: 2025*
