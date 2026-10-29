# ==========================================
# 1. EXPLORAÇÃO DOS DADOS
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento do dataset (Credit Card Fraud Detection)
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Inspeção inicial
print("Dimensões do dataset:", df.shape)
print(df.head())

# Proporção das classes
contagem_fraude = df['Class'].value_counts()
proporcao_fraude = df['Class'].value_counts(normalize=True) * 100

print(f"\nDistribuição das classes:\n{contagem_fraude}")
print(f"\nProporção das classes (%):\n{proporcao_fraude.map('{:.3f}%'.format)}")

# Visualização do desbalanço
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Distribuição de Transações (0: Normal | 1: Fraude)')
plt.yscale('log') # Escala logarítmica para evidenciar a classe minoritária
plt.show()


# ==========================================
# 2. PREPARAÇÃO E ENGENHARIA DE RECURSOS
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Engenharia de atributos: log da variável Amount (para mitigar o impacto de valores discrepantes)
df['Log_Amount'] = np.log1p(df['Amount'])

# Padronização de Time e Log_Amount
scaler = StandardScaler()
df['Scaled_Time'] = scaler.fit_transform(df[['Time']])
df['Scaled_Amount'] = scaler.fit_transform(df[['Log_Amount']])

# Remoção das colunas originais não padronizadas e da variável alvo
X = df.drop(columns=['Class', 'Time', 'Amount', 'Log_Amount'])
y = df['Class']

# Divisão em Treino e Teste com Stratify para preservar a proporção de fraudes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de Treino: {X_train.shape} | Fraudes no treino: {y_train.sum()}")
print(f"Conjunto de Teste:  {X_test.shape}  | Fraudes no teste:  {y_test.sum()}")


# ==========================================
# 3 E 4. TREINAMENTO, AJUSTE E AVALIAÇÃO
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, average_precision_score

# Cálculo do peso da classe para lidar com o desbalanceamento
taxa_desequilibrio = (len(y_train) - sum(y_train)) / sum(y_train)

# Dicionário de modelos
modelos = {
    "Regressão Logística": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(scale_pos_weight=taxa_desequilibrio, random_state=42, eval_metric='logloss')
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Avaliação utilizando o limiar padrão (0.50)
    y_pred = (y_proba >= 0.50).astype(int)
    
    print(f"\n==================== {nome} (Limiar 0.50) ====================")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (Precision-Recall): {average_precision_score(y_test, y_proba):.4f}")
    
    resultados[nome] = {"modelo": modelo, "y_proba": y_proba}

# Curva de Precisão-Recall comparativa
plt.figure(figsize=(8, 6))
for nome, dados in resultados.items():
    precisao, recall, limites = precision_recall_curve(y_test, dados["y_proba"])
    plt.plot(recall, precisao, label=f"{nome}")

plt.xlabel('Recall (Sensibilidade)')
plt.ylabel('Precisão')
plt.title('Curva Precisão-Recall (Foco na Fraude)')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 5. EXPLICABILIDADE COM SHAP
# ==========================================
import shap

# Utilizando o modelo XGBoost para explicabilidade
xgb_modelo = resultados["XGBoost"]["modelo"]
explorador = shap.TreeExplainer(xgb_modelo)
# Amostra de 1000 registros do conjunto de teste para otimizar a performance
valores_shap = explorador(X_test.iloc[:1000])

# Gráfico de impacto geral das variáveis
plt.title("Importância das Variáveis via SHAP")
shap.summary_plot(valores_shap, X_test.iloc[:1000])

# Gráfico individual para a primeira transação fraudulenta encontrada no conjunto de teste
indices_fraude = y_test[y_test == 1].index
exemplo_idx = X_test.index.get_loc(indices_fraude[0])

shap.plots.waterfall(valores_shap[exemplo_idx])