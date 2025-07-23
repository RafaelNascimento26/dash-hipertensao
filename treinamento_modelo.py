# treinamento_modelo.py
# Este arquivo deve ser executado PRIMEIRO para gerar os arquivos .pkl necessários.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
import numpy as np # Importar numpy

print("Iniciando treinamento do modelo e preparação de arquivos para o Dash...")

# === 1. Carregar os dados ===
# Certifique-se de que 'hypertension_dataset.csv' está no mesmo diretório
# onde você executa este script ou forneça o caminho completo.
try:
    df_train = pd.read_csv("hypertension_dataset.csv")
    print("Dataset de treinamento 'hypertension_dataset.csv' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: 'hypertension_dataset.csv' não encontrado. Certifique-se de que está na mesma pasta que treinamento_modelo.py.")
    exit()

# Mapear 'Yes'/'No' para 1/0 para a variável target
df_train["Hypertension_Binary"] = df_train["Has_Hypertension"].map({"Yes": 1, "No": 0})

# === 2. Definir features (variáveis de entrada) e target (variável a prever) ===
X = df_train.drop(columns=["Has_Hypertension", "Hypertension_Binary"])
y = df_train["Hypertension_Binary"]

# === 3. Definir listas de variáveis categóricas e numéricas ===
categorical_features = ["BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"]
numerical_features = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]

# Garantir que valores ausentes em colunas categóricas sejam tratados como string "None"
# antes de passar para o OneHotEncoder, que espera strings.
for col in categorical_features:
    X[col] = X[col].fillna("None").astype(str)

# === 4. Configurar o pré-processamento de colunas com ColumnTransformer ===
# 'cat': aplica OneHotEncoder às colunas categóricas. 'handle_unknown='ignore'' evita erros
# se uma nova categoria aparecer (ela será tratada como zero).
# 'remainder='passthrough'': mantém as colunas numéricas sem alteração.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder="passthrough"
)

# === 5. Criar o Pipeline de Machine Learning ===
# O pipeline encadeia o pré-processamento e o classificador,
# garantindo que as transformações sejam aplicadas de forma consistente.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
])

# === 6. Dividir os dados em conjuntos de treino e teste e treinar o pipeline ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train) # Treina o pipeline completo (pré-processador + classificador)

print("Pipeline de Machine Learning treinado com sucesso!")
print(f"Acurácia do modelo no conjunto de teste: {pipeline.score(X_test, y_test):.2f}")

# === 7. Obter os nomes das colunas após o pré-processamento ===
# Estes nomes serão usados no app Dash para alinhar os valores SHAP com as features corretas.
transformed_column_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
print(f"Total de features após transformação (para SHAP): {len(transformed_column_names)}")
print(f"Primeiras 5 features transformadas: {transformed_column_names[:5]}...")

# === 8. Gerar e salvar um sample de dados de treinamento TRANSFORMADOS para o SHAP Explainer ===
# O SHAP TreeExplainer pode se beneficiar de um 'background_data' para cálculos mais precisos,
# especialmente com One-Hot Encoding. Usamos uma pequena amostra dos dados de treino já transformados.
# É crucial que X_train tenha os mesmos nomes de coluna que X, antes da transformação.
X_train_transformed_sample = pipeline.named_steps['preprocessor'].transform(X_train.sample(n=100, random_state=42))
print(f"Shape do sample transformado para background do SHAP: {X_train_transformed_sample.shape}")


# === 9. Salvar todos os arquivos .pkl necessários para o aplicativo Dash ===
# Eles serão salvos na mesma pasta onde este script é executado.
output_model_path = "model.pkl"
output_cols_path = "transformed_column_names.pkl"
output_shap_background_path = "shap_background_data.pkl"

with open(output_model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Arquivo '{output_model_path}' (pipeline completo) salvo com sucesso!")

with open(output_cols_path, "wb") as f:
    pickle.dump(transformed_column_names, f)
print(f"Arquivo '{output_cols_path}' (nomes das colunas transformadas) salvo com sucesso!")

with open(output_shap_background_path, "wb") as f:
    pickle.dump(X_train_transformed_sample, f)
print(f"Arquivo '{output_shap_background_path}' (dados de background para SHAP) salvo com sucesso!")

print("\nTreinamento e salvamento de arquivos .pkl concluídos com sucesso.")
print("Agora você pode executar 'main.py' para iniciar o aplicativo Dash.")