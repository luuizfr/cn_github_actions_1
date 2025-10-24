# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Leitura dos dados
df = pd.read_csv("data/sample.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 2. Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinamento do modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)

# 5. Cálculo das métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 6. Geração do relatório detalhado
report = classification_report(y_test, y_pred)

# 7. Salvamento do relatório com métricas principais
with open("report.txt", "w", encoding="utf-8") as f:
    f.write("=== RELATÓRIO DE CLASSIFICAÇÃO ===\n\n")
    f.write(f"Acurácia: {accuracy:.4f}\n")
    f.write(f"Precisão: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")
    f.write("=== RELATÓRIO DETALHADO ===\n")
    f.write(report)
    f.write(f"\n\n=== INFORMAÇÕES DO DATASET ===\n")
    f.write(f"Total de amostras: {len(df)}\n")
    f.write(f"Amostras de treino: {len(X_train)}\n")
    f.write(f"Amostras de teste: {len(X_test)}\n")
    f.write(f"Classes: {sorted(y.unique())}\n")

print("Treinamento concluído! Relatório salvo em report.txt")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
