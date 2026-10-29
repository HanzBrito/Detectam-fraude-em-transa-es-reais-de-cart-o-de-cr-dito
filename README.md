# Detecção de Fraudes em Cartões de Crédito com Machine Learning Explicável

## 1. O Problema do Desbalanceamento e a Métrica Ideal
Em bases de detecção de fraudes bancárias, apenas cerca de **0,17%** das transações representam fraudes reais. 
- **O problema da acurácia:** Um modelo ingênuo que preveja "não é fraude" para absolutamente todas as transações atinge **99,83% de acurácia**, mas falha em detectar 100% dos golpes, sendo completamente inútil para a instituição financeira.
- **Métricas foco:** 
  - **Recall (Sensibilidade):** Mede a capacidade de encontrar todas as fraudes reais (essencial para evitar prejuízos).
  - **Precisão:** Mede quantos dos alertas gerados eram de fato fraudes (essencial para evitar bloqueios indevidos de clientes legítimos).
  - **PR-AUC (Area Under the Precision-Recall Curve):** Métrica robusta para avaliar modelos em cenários altamente desbalanceados.

## 2. Preparação dos Dados
- **Engenharia de Atributos:** Aplicação de transformação logarítmica (`np.log1p`) na coluna `Amount` para suavizar a cauda longa de valores monetários.
- **Normalização:** Utilização do `StandardScaler` nas colunas de tempo (`Time`) e valor transformado (`Log_Amount`).
- **Estratificação:** A divisão entre treino e teste utilizou `stratify=y` para assegurar que a proporção exata de fraudes fosse mantida em ambos os subconjuntos.

## 3. Comparação de Modelos (Classe Minoritária - Fraude)

| Modelo | Recall | Precisão | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Regressão Logística | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- |
| XGBoost | -- | -- | -- | -- |

*(Substitua os traços pelas métricas obtidas na execução do seu notebook)*

## 4. Limiar de Decisão e Explicabilidade (SHAP)
- **Ajuste de Limiar:** Modificar o limiar de classificação de `0.50` para patamares inferiores (ex: `0.30`) permite flexibilizar o modelo para capturar mais fraudes, aumentando o Recall.
- **Explicabilidade:** O uso da biblioteca SHAP revelou que variáveis específicas geradas por PCA (como `V14`, `V12` e `V10`) exercem o peso mais crítico na tomada de decisão dos algoritmos baseados em árvores.

## 5. Diferenciais e Evolução do Projeto
- Abordagem direcionada a resolver o problema de negócio e não apenas otimizar métricas genéricas de acurácia.
- Implementação de gráficos explicativos detalhados (`summary_plot` e `waterfall` do SHAP) para auditoria individual de transações suspeitas.