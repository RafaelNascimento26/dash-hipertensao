from flask import Flask, render_template_string
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import plotly.express as px
import os
import shap
import numpy as np
import plotly.graph_objects as go
import datetime

# --- CONFIGURAÇÃO DO FLASK PRINCIPAL ---
server = Flask(__name__)  # Inicializa a aplicação Flask principal

# --- INTEGRAÇÃO DO DASH NO FLASK ---
# O prefixo de requisição é onde seu Dash app será montado dentro do Flask
DASH_APP_ROUTE_PREFIX = '/dash_hypertension/'
app = dash.Dash(__name__, server=server, url_base_pathname=DASH_APP_ROUTE_PREFIX,
                external_stylesheets=[dbc.themes.FLATLY,
                                      "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"])
app.title = "Previsão de Hipertensão"

# Certifique-se de que o Dash não anexe as rotas no momento da inicialização
# Isso é importante para evitar conflitos se você tiver outras rotas no Flask
app.config.suppress_callback_exceptions = True

# === Configuração do Banco de Dados (CSV) para Pacientes Simulados ===
SIMULATED_PATIENTS_DB = "simulated_patients.csv"
SIMULATED_PATIENTS_COLS = [
    "Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI",
    "BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status",
    "Predicted_Hypertension", "Prediction_Probability", "Timestamp"
]


def initialize_simulated_patients_db():
    """
    Inicializa o arquivo CSV do banco de dados de pacientes simulados
    se ele não existir, garantindo que todas as colunas necessárias estejam presentes.
    """
    if not os.path.exists(SIMULATED_PATIENTS_DB):
        df_empty = pd.DataFrame(columns=SIMULATED_PATIENTS_COLS)
        df_empty.to_csv(SIMULATED_PATIENTS_DB, index=False)
        print(f"Arquivo de banco de dados '{SIMULATED_PATIENTS_DB}' criado com cabeçalhos.")
    else:
        # Verifica se o arquivo existente possui todas as colunas
        try:
            df_existing = pd.read_csv(SIMULATED_PATIENTS_DB)
            missing_cols = [col for col in SIMULATED_PATIENTS_COLS if col not in df_existing.columns]
            if missing_cols:
                print(f"ATENÇÃO: '{SIMULATED_PATIENTS_DB}' existe, mas faltam as colunas: {missing_cols}.")
                print("Recomendado: Apagar o arquivo e rodar novamente para recriá-lo com todas as colunas.")
            else:
                print(f"Arquivo de banco de dados '{SIMULATED_PATIENTS_DB}' já existe e possui as colunas esperadas.")
        except Exception as e:
            print(f"ERRO ao ler '{SIMULATED_PATIENTS_DB}'. Pode estar corrompido ou ter formato inesperado: {e}")
            print("Recomendado: Apagar o arquivo e rodar novamente para recriá-lo.")


initialize_simulated_patients_db()

# === 1. Carregar modelo e dados ===
MODEL_PIPELINE = None
TRANSFORMED_COLUMN_NAMES = []
SHAP_BACKGROUND_DATA = None
PREPROCESSOR = None
CLASSIFIER = None

try:
    with open("model.pkl", "rb") as f:
        MODEL_PIPELINE = pickle.load(f)
    print("Modelo (pipeline) carregado com sucesso!")
    PREPROCESSOR = MODEL_PIPELINE.named_steps['preprocessor']
    CLASSIFIER = MODEL_PIPELINE.named_steps['classifier']
except FileNotFoundError:
    print(
        "ERRO: 'model.pkl' não encontrado. Certifique-se de que o arquivo está na mesma pasta ou que o caminho está correto.")
    print("A aplicação pode não funcionar corretamente sem o modelo.")
except KeyError as e:
    print(
        f"ERRO: Componente esperado no pipeline não encontrado: {e}. Verifique se 'preprocessor' e 'classifier' existem.")
    print("A aplicação pode não funcionar corretamente sem os componentes do pipeline.")
except Exception as e:
    print(f"ERRO ao carregar o modelo (pipeline): {e}")
    print("A aplicação pode não funcionar corretamente sem o modelo.")

try:
    TRANSFORMED_COLUMN_NAMES = pickle.load(open("transformed_column_names.pkl", "rb"))
    print("Nomes das colunas transformadas carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: 'transformed_column_names.pkl' não encontrado. Rode 'treinamento_modelo.py' PRIMEIRO.")
    print("A interpretabilidade SHAP pode não funcionar corretamente.")
except Exception as e:
    print(f"ERRO ao carregar nomes das colunas transformadas: {e}")
    print("A interpretabilidade SHAP pode não funcionar corretamente.")

try:
    SHAP_BACKGROUND_DATA = pickle.load(open("shap_background_data.pkl", "rb"))
    print("SHAP background data carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: 'shap_background_data.pkl' não encontrado. Rode 'treinamento_modelo.py' PRIMEIRO.")
    print("A interpretabilidade SHAP pode não funcionar corretamente.")
except Exception as e:
    print(f"ERRO ao carregar SHAP background data: {e}")
    print("A interpretabilidade SHAP pode não funcionar corretamente.")

ORIGINAL_MODEL_FEATURES = [
    "Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI",
    "BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"
]

categorical_cols = ["BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"]
numerical_features = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]


# === Funções Auxiliares para Carregar, Processar Dados e Gerar Gráficos/Estatísticas ===
def load_and_process_data():
    """
    Carrega e processa os datasets original e simulado, combinando-os para gráficos.
    Garante que as colunas necessárias existam para evitar KeyErrors.
    """
    df_original = pd.DataFrame(columns=[col for col in ORIGINAL_MODEL_FEATURES] + ['Has_Hypertension'])
    try:
        df_original_loaded = pd.read_csv("hypertension_dataset.csv")
        for col in df_original_loaded.columns:
            if col in df_original.columns:
                df_original[col] = df_original_loaded[col]
            elif col == 'Has_Hypertension':
                df_original['Has_Hypertension'] = df_original_loaded['Has_Hypertension']
    except FileNotFoundError:
        print("AVISO: 'hypertension_dataset.csv' não encontrado. Gráficos com dados originais estarão vazios.")
    except Exception as e:
        print(f"ERRO ao carregar 'hypertension_dataset.csv': {e}")
        print("Gráficos com dados originais podem estar incompletos ou vazios.")

    df_simulated = pd.DataFrame(columns=SIMULATED_PATIENTS_COLS)
    try:
        df_simulated_loaded = pd.read_csv(SIMULATED_PATIENTS_DB)
        for col in df_simulated_loaded.columns:
            if col in df_simulated.columns:
                df_simulated[col] = df_simulated_loaded[col]
    except FileNotFoundError:
        print("AVISO: 'simulated_patients.csv' não encontrado (ou vazio). Pacientes simulados estarão vazios.")
    except pd.errors.EmptyDataError:
        print("AVISO: 'simulated_patients.csv' está vazio. Não há pacientes simulados.")
    except Exception as e:
        print(f"ERRO ao carregar 'simulated_patients.csv': {e}")
        print("Pacientes simulados podem estar incompletos ou vazios.")

    for col in categorical_cols:
        if col in df_original.columns:
            df_original[col] = df_original[col].fillna("None").astype(str)
        if col in df_simulated.columns:
            df_simulated[col] = df_simulated[col].fillna("None").astype(str)

    if 'Has_Hypertension' in df_original.columns and not df_original['Has_Hypertension'].empty:
        df_original['Hypertension_Status'] = df_original['Has_Hypertension'].map(
            {'Yes': 'Hipertenso', 'No': 'Não Hipertenso'})
        df_original['Hypertension_Binary_Numeric'] = df_original['Has_Hypertension'].map({'Yes': 1, 'No': 0})
    else:
        df_original['Hypertension_Status'] = pd.Series(dtype='str')
        df_original['Hypertension_Binary_Numeric'] = pd.Series(dtype='float')

    if 'Predicted_Hypertension' in df_simulated.columns and not df_simulated['Predicted_Hypertension'].empty:
        df_simulated['Hypertension_Status'] = df_simulated['Predicted_Hypertension']
        df_simulated['Hypertension_Binary_Numeric'] = df_simulated['Predicted_Hypertension'].map(
            {'Hipertenso': 1, 'Não Hipertenso': 0})
    else:
        df_simulated['Hypertension_Status'] = pd.Series(dtype='str')
        df_simulated['Hypertension_Binary_Numeric'] = pd.Series(dtype='float')

    df_original['Source'] = 'Original Dataset'
    df_simulated['Source'] = 'Simulated Patients'

    common_graph_cols = list(set(numerical_features + categorical_cols +
                                 ['Hypertension_Status', 'Hypertension_Binary_Numeric', 'Source']))

    df_original_filtered = df_original[[col for col in common_graph_cols if col in df_original.columns]]
    df_simulated_filtered = df_simulated[[col for col in common_graph_cols if col in df_simulated.columns]]

    df_for_graphs = pd.concat([df_original_filtered, df_simulated_filtered], ignore_index=True)

    for col in numerical_features + ['Hypertension_Binary_Numeric']:
        if col in df_for_graphs.columns:
            df_for_graphs[col] = pd.to_numeric(df_for_graphs[col], errors='coerce')

    return df_original, df_simulated, df_for_graphs


def get_dashboard_stats(df_original, df_simulated, df_for_graphs):
    """Calcula e retorna estatísticas chave para o dashboard."""
    total_original = len(df_original)
    total_simulated = len(df_simulated)
    total_combined = len(df_for_graphs)

    original_hypertension_counts = df_original['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'], fill_value=0) if 'Hypertension_Status' in df_original.columns else pd.Series(
        {'Hipertenso': 0, 'Não Hipertenso': 0})
    simulated_hypertension_counts = df_simulated['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'], fill_value=0) if 'Hypertension_Status' in df_simulated.columns else pd.Series(
        {'Hipertenso': 0, 'Não Hipertenso': 0})
    combined_hypertension_counts = df_for_graphs['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'],
        fill_value=0) if 'Hypertension_Status' in df_for_graphs.columns else pd.Series(
        {'Hipertenso': 0, 'Não Hipertenso': 0})

    return [
        f"Total de pacientes: {total_original}",
        f"Hipertensos: {original_hypertension_counts['Hipertenso']}",
        f"Não Hipertensos: {original_hypertension_counts['Não Hipertenso']}",
        f"Total de pacientes: {total_simulated}",
        f"Hipertensos: {simulated_hypertension_counts['Hipertenso']}",
        f"Não Hipertensos: {simulated_hypertension_counts['Não Hipertenso']}",
        f"Total de pacientes: {total_combined}",
        f"Total Hipertensos: {combined_hypertension_counts['Hipertenso']}",
        f"Total Não Hipertensos: {combined_hypertension_counts['Não Hipertenso']}",
    ]


def generate_all_graphs(df_for_graphs):
    """Gera todos os objetos de gráfico Plotly baseados nos dados combinados."""
    age_in_df = "Age" in df_for_graphs.columns and not df_for_graphs["Age"].isnull().all()
    bmi_in_df = "BMI" in df_for_graphs.columns and not df_for_graphs["BMI"].isnull().all()
    salt_in_df = "Salt_Intake" in df_for_graphs.columns and not df_for_graphs["Salt_Intake"].isnull().all()
    stress_in_df = "Stress_Score" in df_for_graphs.columns and not df_for_graphs["Stress_Score"].isnull().all()
    bp_history_in_df = "BP_History" in df_for_graphs.columns and not df_for_graphs["BP_History"].isnull().all()
    smoking_status_in_df = "Smoking_Status" in df_for_graphs.columns and not df_for_graphs[
        "Smoking_Status"].isnull().all()
    hypertension_status_in_df = "Hypertension_Status" in df_for_graphs.columns and not df_for_graphs[
        "Hypertension_Status"].isnull().all()
    hypertension_binary_numeric_in_df = "Hypertension_Binary_Numeric" in df_for_graphs.columns and not df_for_graphs[
        "Hypertension_Binary_Numeric"].isnull().all()

    all_numerical_and_target_present_for_heatmap = all(
        col in df_for_graphs.columns and not df_for_graphs[col].isnull().all() for col in
        numerical_features + ['Hypertension_Binary_Numeric'])
    all_numerical_and_status_present_for_scatter_matrix = all(
        col in df_for_graphs.columns and not df_for_graphs[col].isnull().all() for col in
        numerical_features + ['Hypertension_Status'])

    scatter_matrix_fig = {}
    if all_numerical_and_status_present_for_scatter_matrix and not df_for_graphs[numerical_features].dropna().empty:
        scatter_matrix_fig = px.scatter_matrix(
            df_for_graphs.dropna(subset=numerical_features + ['Hypertension_Status']),
            dimensions=numerical_features,
            color="Hypertension_Status",
            title="Matriz de Dispersão das Variáveis Numéricas por Status de Hipertensão",
            template="plotly_white",
            color_discrete_map={"Hipertenso": "red", "Não Hipertenso": "green"})
        scatter_matrix_fig.update_yaxes(tickangle=0, automargin=True)

    return [
        px.histogram(df_for_graphs, x="Age", nbins=30, title="Distribuição de Idade", template="plotly_white",
                     color='Source', barmode='overlay') if age_in_df else {},
        px.histogram(df_for_graphs, x="BMI", nbins=30, title="Distribuição de IMC", template="plotly_white",
                     color='Source', barmode='overlay') if bmi_in_df else {},
        px.box(df_for_graphs, x="Hypertension_Status", y="Age", color="Hypertension_Status",
               title="Idade vs. Hipertensão", template="plotly_white", points="outliers",
               color_discrete_map={"Hipertenso": "red",
                                   "Não Hipertenso": "green"}) if age_in_df and hypertension_status_in_df else {},
        px.box(df_for_graphs, x="Hypertension_Status", y="Salt_Intake", color="Hypertension_Status",
               title="Consumo de Sal vs. Hipertensão", template="plotly_white", points="outliers",
               color_discrete_map={"Hipertenso": "red",
                                   "Não Hipertenso": "green"}) if salt_in_df and hypertension_status_in_df else {},
        px.bar(df_for_graphs.groupby(['BP_History', 'Hypertension_Status']).size().reset_index(name='count'),
               x='BP_History', y='count', color='Hypertension_Status', title='Histórico de Pressão vs. Hipertensão',
               template="plotly_white", barmode='group', color_discrete_map={"Hipertenso": "red",
                                                                             "Não Hipertenso": "green"}) if bp_history_in_df and hypertension_status_in_df else {},
        px.bar(df_for_graphs.groupby(['Smoking_Status', 'Hypertension_Status']).size().reset_index(name='count'),
               x='Smoking_Status', y='count', color='Hypertension_Status', title='Status de Fumante vs. Hipertensão',
               template="plotly_white", barmode='group', color_discrete_map={"Hipertenso": "red",
                                                                             "Não Hipertenso": "green"}) if smoking_status_in_df and hypertension_status_in_df else {},
        px.imshow(df_for_graphs[numerical_features + ['Hypertension_Binary_Numeric']].corr(), text_auto=True,
                  color_continuous_scale='RdBu_r', title='Correlação entre Variáveis Numéricas e Hipertensão',
                  template="plotly_white") if all_numerical_and_target_present_for_heatmap else {},
        px.violin(df_for_graphs, x="Hypertension_Status", y="Stress_Score", color="Hypertension_Status",
                  title="Distribuição do Estresse vs. Hipertensão", template="plotly_white", points="all", box=True,
                  color_discrete_map={"Hipertenso": "red",
                                      "Não Hipertenso": "green"}) if stress_in_df and hypertension_status_in_df else {},
        scatter_matrix_fig,
    ]


# Carregar dados e gerar estatísticas/gráficos para o layout inicial (usados no Dash App)
df_original_initial, df_simulated_initial, df_for_graphs_initial = load_and_process_data()
initial_stats_output = get_dashboard_stats(df_original_initial, df_simulated_initial, df_for_graphs_initial)
initial_graphs_output = generate_all_graphs(df_for_graphs_initial)

# === DEFINIÇÕES DAS OPÇÕES DOS DROPDOWNS ===
bp_options = ["Normal", "Hypertension", "Prehypertension"]
medication_options = ["None", "ACE Inhibitor", "Other", "Beta Blocker", "Diuretic"]
family_options = ["Yes", "No"]
exercise_options = ["Low", "Moderate", "High"]
smoking_options = ["Smoker", "Non-Smoker"]


# === Funções para Recomendações (mantida a mesma) ===
def generate_prevention_recommendations(input_data, prediction_status):
    recommendations = []

    if prediction_status == "Hipertenso":
        recommendations.append(html.Li([
            html.I(className="bi bi-exclamation-octagon-fill text-danger me-2"),
            html.Span("Atenção! Sua previsão indica ", className="fw-bold"),
            html.Span("risco de hipertensão.", className="text-danger fw-bold")
        ], className="mb-2"))
        recommendations.append(html.Li([
            html.I(className="bi bi-hospital-fill text-info me-2"),
            "É fundamental ", html.B("procurar um médico"),
            " para um diagnóstico preciso e acompanhamento. Não se automedique e siga as orientações profissionais."
        ], className="mb-2"))
    else:
        recommendations.append(html.Li([
            html.I(className="bi bi-check-circle-fill text-success me-2"),
            html.Span("Excelente! Sua previsão atual indica ", className="fw-bold"),
            html.Span("baixo risco de hipertensão.", className="text-success fw-bold")
        ], className="mb-2"))
        recommendations.append(html.Li([
            html.I(className="bi bi-award-fill text-primary me-2"),
            "Mantenha seus hábitos saudáveis para continuar protegendo sua saúde cardiovascular."
        ], className="mb-2"))

    if input_data.get("Salt_Intake") is not None and input_data["Salt_Intake"] > 7:
        recommendations.append(html.Li([
            html.I(className="bi bi-exclamation-triangle-fill text-warning me-2"),
            "O consumo excessivo de sal (", html.B(f"{input_data['Salt_Intake']}g/dia"),
            ") é um fator de risco significativo. Tente reduzir a ingestão de alimentos processados e use menos sal ao cozinhar."
        ], className="mb-2"))
    elif input_data.get("Salt_Intake") is not None and input_data["Salt_Intake"] > 5 and input_data["Salt_Intake"] <= 7:
        recommendations.append(html.Li([
            html.I(className="bi bi-info-circle-fill text-info me-2"),
            "Seu consumo de sal (", html.B(f"{input_data['Salt_Intake']}g/dia"),
            ") está moderado. Pequenas reduções podem trazer benefícios adicionais para a pressão arterial."
        ], className="mb-2"))

    if input_data.get("Stress_Score") is not None and input_data["Stress_Score"] > 7:
        recommendations.append(html.Li([
            html.I(className="bi bi-emoji-frown-fill text-warning me-2"),
            "Níveis elevados de estresse (", html.B(f"pontuação {input_data['Stress_Score']}"),
            ") podem impactar a pressão arterial. Considere técnicas de relaxamento como meditação, yoga ou exercícios leves."
        ], className="mb-2"))
    elif input_data.get("Stress_Score") is not None and input_data["Stress_Score"] >= 5 and input_data[
        "Stress_Score"] <= 7:
        recommendations.append(html.Li([
            html.I(className="bi bi-bell-fill text-info me-2"),
            "Gerenciar o estresse é importante para sua saúde geral. Reserve um tempo para atividades que você gosta."
        ], className="mb-2"))

    if input_data.get("Sleep_Duration") is not None and input_data["Sleep_Duration"] < 6.5:
        recommendations.append(html.Li([
            html.I(className="bi bi-moon-fill text-warning me-2"),
            "A falta de sono de qualidade (", html.B(f"{input_data['Sleep_Duration']}h"),
            ") afeta a saúde cardiovascular. Tente estabelecer uma rotina de sono e garanta 7-9 horas por noite."
        ], className="mb-2"))
    elif input_data.get("Sleep_Duration") is not None and input_data["Sleep_Duration"] > 8.5:
        recommendations.append(html.Li([
            html.I(className="bi bi-hourglass-split text-info me-2"),
            "A duração ideal do sono varia, mas mais de 8.5 horas pode estar associado a outros fatores. Mantenha um sono de qualidade e consulte um profissional se tiver preocupações."
        ], className="mb-2"))

    if input_data.get("BMI") is not None and input_data["BMI"] > 25:
        recommendations.append(html.Li([
            html.I(className="bi bi-person-bounding-box text-danger me-2"),
            "Manter um peso saudável é crucial. Com IMC de ", html.B(f"{input_data['BMI']}"),
            ", perder peso, mesmo que moderadamente, pode reduzir significativamente o risco de hipertensão."
        ], className="mb-2"))
    elif input_data.get("BMI") is not None and input_data["BMI"] < 18.5:
        recommendations.append(html.Li([
            html.I(className="bi bi-person-dash-fill text-info me-2"),
            "Verifique se seu IMC (", html.B(f"{input_data['BMI']}"),
            ") está adequado para sua altura. Um peso muito baixo também pode indicar outros problemas de saúde."
        ], className="mb-2"))

    if input_data.get("BP_History") is not None and input_data["BP_History"] in ["Hypertension", "Prehypertension"]:
        recommendations.append(html.Li([
            html.I(className="bi bi-heart-pulse-fill text-danger me-2"),
            "Seu histórico de pressão arterial (", html.B(f"{input_data['BP_History']}"),
            ") exige acompanhamento médico regular e adesão ao tratamento, se indicado."
        ], className="mb-2"))
    elif input_data.get("BP_History") is not None and input_data["BP_History"] == "Normal":
        recommendations.append(html.Li([
            html.I(className="bi bi-patch-check-fill text-success me-2"),
            "Seu histórico de pressão arterial é normal, o que é ótimo! Mantenha exames regulares para monitoramento contínuo."
        ], className="mb-2"))

    if input_data.get("Medication") is not None and input_data["Medication"] != "None":
        recommendations.append(html.Li([
            html.I(className="bi bi-capsule-fill text-info me-2"),
            "A medicação atual (", html.B(f"{input_data['Medication']}"),
            ") é parte importante do seu manejo de saúde. Continue seguindo as orientações médicas rigorosamente."
        ], className="mb-2"))

    if input_data.get("Family_History") is not None and input_data["Family_History"] == "Yes":
        recommendations.append(html.Li([
            html.I(className="bi bi-people-fill text-warning me-2"),
            "Com histórico familiar de hipertensão (", html.B("Sim"),
            "), você tem um risco aumentado. Adotar um estilo de vida saudável e fazer check-ups regulares é ainda mais importante."
        ], className="mb-2"))
    else:
        if input_data.get("Family_History") is not None and input_data["Family_History"] == "No":
            recommendations.append(html.Li([
                html.I(className="bi bi-people-fill text-success me-2"),
                "Seu histórico familiar para hipertensão (", html.B("Não"),
                ") é um ponto positivo, mas a prevenção ativa ainda é essencial."
            ], className="mb-2"))

    if input_data.get("Exercise_Level") is not None and input_data["Exercise_Level"] == "Low":
        recommendations.append(html.Li([
            html.I(className="bi bi-bicycle text-warning me-2"),
            "Seu nível de atividade física (", html.B("Baixo"),
            ") é uma área para melhoria. Comece com 30 minutos de caminhada moderada na maioria dos dias da semana."
        ], className="mb-2"))
    elif input_data.get("Exercise_Level") is not None and input_data["Exercise_Level"] == "Moderate":
        recommendations.append(html.Li([
            html.I(className="bi bi-person-walking text-info me-2"),
            "Seu nível de exercício é (", html.B("Moderado"),
            "). Tente incorporar atividades de maior intensidade ou variar sua rotina para mais benefícios."
        ], className="mb-2"))
    elif input_data.get("Exercise_Level") is not None and input_data["Exercise_Level"] == "High":
        recommendations.append(html.Li([
            html.I(className="bi bi-trophy-fill text-success me-2"),
            "Ótimo trabalho com seu nível de exercício (", html.B("Alto"),
            ")! Continue assim para manter sua saúde cardiovascular."
        ], className="mb-2"))

    if input_data.get("Smoking_Status") is not None and input_data["Smoking_Status"] == "Smoker":
        recommendations.append(html.Li([
            html.I(className="bi bi-x-octagon-fill text-danger me-2"),
            html.B("Parar de fumar"),
            " é uma das ações mais impactantes para prevenir e controlar a hipertensão, além de melhorar sua saúde geral."
        ], className="mb-2"))
    else:
        if input_data.get("Smoking_Status") is not None and input_data["Smoking_Status"] == "Non-Smoker":
            recommendations.append(html.Li([
                html.I(className="bi bi-fire text-success me-2"),
                "Seu status de não fumante (", html.B("Não Fumante"),
                ") é um fator protetor importante para sua saúde cardiovascular."
            ], className="mb-2"))

    recommendations.append(html.Li([
        html.I(className="bi bi-file-earmark-medical-fill text-primary me-2"),
        html.B("Lembre-se:"),
        " estas recomendações são geradas por inteligência artificial com base nos dados fornecidos e no modelo. ",
        html.B("Consulte sempre um profissional de saúde qualificado para aconselhamento e tratamento personalizados.")
    ], className="mt-4 mb-2 p-2 bg-light border-start border-primary border-5 rounded"))

    return html.Ul(recommendations, className="list-unstyled p-0")


# === Layout do Aplicativo Dash (seu dashboard atual) ===
app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.Div([
            html.H1("🩺 Dashboard de Previsão de Hipertensão com ML", className="display-4 fw-bold text-center mb-0"),
            html.P(
                "Bem-vindo(a) ao Dashboard de Previsão de Hipertensão! Este aplicativo utiliza um modelo de Machine Learning para estimar o risco de um paciente desenvolver hipertensão com base em diversos fatores. Explore os gráficos para entender os dados e use a simulação para testar diferentes cenários.",
                className="lead text-center text-muted"),
        ], className="py-4 px-3 bg-light rounded shadow-sm mb-5"),
    )),

    dbc.Card(
        dbc.CardBody([
            html.H3("Estatísticas Chave do Dataset", className="card-title text-center mb-4"),
            html.P(
                "Esta seção apresenta um resumo quantitativo dos pacientes no dataset original, dos pacientes que você simulou e da combinação de ambos.",
                className="text-muted text-center mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Dataset Original", className="card-title text-center text-primary"),
                        html.P(initial_stats_output[0], id="total-original-patients",
                               className="card-text text-center"),
                        html.P(initial_stats_output[1], id="original-hypertensive-patients",
                               className="card-text text-center text-danger"),
                        html.P(initial_stats_output[2], id="original-non-hypertensive-patients",
                               className="card-text text-center text-success"),
                    ]), className="h-100 shadow-sm"), lg=4, md=12, className="mb-3"),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Dataset Novo", className="card-title text-center text-primary"),
                        html.P(initial_stats_output[3], id="total-simulated-patients",
                               className="card-text text-center"),
                        html.P(initial_stats_output[4], id="simulated-hypertensive-patients",
                               className="card-text text-center text-danger"),
                        html.P(initial_stats_output[5], id="simulated-non-hypertensive-patients",
                               className="card-text text-center text-success"),
                    ]), className="h-100 shadow-sm"), lg=4, md=12, className="mb-3"),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Dataset Combinado", className="card-title text-center text-primary"),
                        html.P(initial_stats_output[6], id="total-combined-patients",
                               className="card-text text-center"),
                        html.P(initial_stats_output[7], id="combined-hypertensive-patients",
                               className="card-text text-center text-danger"),
                        html.P(initial_stats_output[8], id="combined-non-hypertensive-patients",
                               className="card-text text-center text-success"),
                    ]), className="h-100 shadow-sm"), lg=4, md=12, className="mb-3"),
            ]),
        ]),
        className="mb-5 shadow-sm border-0"
    ),
    html.Hr(className="my-5"),

    dbc.Card(
        dbc.CardBody([
            html.H3("Visão Geral dos Dados", className="card-title text-center mb-4"),
            html.P(
                "Nesta seção, você pode observar a distribuição das principais variáveis do dataset. Os gráficos incluem tanto os dados originais de treinamento quanto os pacientes que você simulou, permitindo uma visão mais completa.",
                className="text-muted text-center mb-4"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    id="graph-age-distribution",
                    figure=initial_graphs_output[0],
                    config={'displayModeBar': False}
                ), lg=6, md=12),

                dbc.Col(dcc.Graph(
                    id="graph-bmi-distribution",
                    figure=initial_graphs_output[1],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
            ]),
            html.Hr(className="my-4"),

            html.H3("Relação entre Variáveis e Hipertensão", className="card-title text-center mt-5 mb-4"),
            html.P(
                "Estes gráficos detalham como cada fator se relaciona com a presença de hipertensão no dataset combinado (original + simulado). Observe as diferenças nas distribuições e contagens para pacientes com e sem a condição. Você pode ver como suas simulações se comparam aos dados reais.",
                className="text-muted text-center mb-4"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    id="graph-age-hypertension",
                    figure=initial_graphs_output[2],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
                dbc.Col(dcc.Graph(
                    id="graph-salt-hypertension",
                    figure=initial_graphs_output[3],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    id="graph-bp-history-hypertension",
                    figure=initial_graphs_output[4],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
                dbc.Col(dcc.Graph(
                    id="graph-smoking-hypertension",
                    figure=initial_graphs_output[5],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
            ]),
            dbc.Row([  # Nova linha para o Heatmap e Violin Plot
                dbc.Col(dcc.Graph(
                    id="graph-correlation-heatmap",
                    figure=initial_graphs_output[6],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
                dbc.Col(dcc.Graph(
                    id="graph-stress-hypertension",
                    figure=initial_graphs_output[7],
                    config={'displayModeBar': False}
                ), lg=6, md=12),
            ]),
            dbc.Row([  # NOVA LINHA PARA O SCATTER PLOT MATRIX
                dbc.Col(dcc.Graph(
                    id="graph-scatter-matrix",
                    figure=initial_graphs_output[8],
                    config={'displayModeBar': False}
                ), width=12),
            ]),
        ]),
        className="mb-5 shadow-sm border-0"
    ),

    html.Hr(className="my-5"),

    dbc.Card(
        dbc.CardBody([
            html.H3("🔍 Insira Dados do Paciente", className="card-title text-center mb-4"),
            html.P(
                "Insira os dados de um paciente abaixo para que o modelo de Machine Learning possa prever o risco de hipertensão. Use os campos de entrada e selecione as opções nos menus suspensos. A previsão e uma explicação dos fatores mais influentes serão exibidos abaixo. Os dados desta simulação serão automaticamente armazenados, **contribuindo para análises futuras e para o aprimoramento contínuo da compreensão sobre a hipertensão.**",
                className="text-muted text-center mb-4"
            ),
            dbc.Row([
                dbc.Col([
                    html.H5("Dados Demográficos e Hábitos", className="mb-3 text-primary"),
                    dbc.Label("Idade"),
                    dbc.Input(id="input-age", type="number", value=45, min=1, max=120, placeholder="Idade (1-120)",
                              className="mb-2"),
                    html.Div(id="error-age", className="text-danger small mb-3"),

                    dbc.Label("Consumo de Sal (g/dia)"),
                    dbc.Input(id="input-sal", type="number", value=9, min=0, max=20, placeholder="Sal (0-20g)",
                              className="mb-2"),
                    html.Div(id="error-sal", className="text-danger small mb-3"),

                    dbc.Label("Estresse (0-10)"),
                    dbc.Input(id="input-estresse", type="number", value=6, min=0, max=10, placeholder="Estresse (0-10)",
                              className="mb-2"),
                    html.Div(id="error-estresse", className="text-danger small mb-3"),

                    dbc.Label("Horas de Sono"),
                    dbc.Input(id="input-sono", type="number", value=6.5, min=0, max=24, step=0.5,
                              placeholder="Sono (0-24h)", className="mb-2"),
                    html.Div(id="error-sono", className="text-danger small mb-3"),

                    dbc.Label("IMC"),
                    dbc.Input(id="input-bmi", type="number", value=27, min=10, max=80, step=0.1,
                              placeholder="IMC (10-80)", className="mb-2"),
                    html.Div(id="error-bmi", className="text-danger small mb-3"),
                ], lg=6, md=12, className="mb-4"),

                dbc.Col([
                    html.H5("Histórico e Estilo de Vida", className="mb-3 text-primary"),
                    dbc.Label("Histórico de Pressão"),
                    dcc.Dropdown(id="input-bp", options=[{"label": i, "value": i} for i in bp_options], value="Normal",
                                 clearable=False, className="mb-2"),
                    html.Div(id="error-bp", className="text-danger small mb-3"),

                    dbc.Label("Medicação"),
                    dcc.Dropdown(id="input-med", options=[{"label": i, "value": i} for i in medication_options],
                                 value="None", clearable=False, className="mb-2"),
                    html.Div(id="error-med", className="text-danger small mb-3"),

                    dbc.Label("Histórico Familiar"),
                    dcc.Dropdown(id="input-fam", options=[{"label": i, "value": i} for i in family_options],
                                 value="Yes", clearable=False, className="mb-2"),
                    html.Div(id="error-fam", className="text-danger small mb-3"),

                    dbc.Label("Atividade Física"),
                    dcc.Dropdown(id="input-ex", options=[{"label": i, "value": i} for i in exercise_options],
                                 value="Low", clearable=False, className="mb-2"),
                    html.Div(id="error-ex", className="text-danger small mb-3"),

                    dbc.Label("Fumante"),
                    dcc.Dropdown(id="input-smk", options=[{"label": i, "value": i} for i in smoking_options],
                                 value="Smoker", clearable=False, className="mb-2"),
                    html.Div(id="error-smk", className="text-danger small mb-3"),
                ], lg=6, md=12, className="mb-4"),
            ]),

            dbc.Button("Prever Risco", id="btn-predict", color="primary", size="lg", className="mt-3 d-grid w-100"),

            html.Div(id="db-save-status", className="text-center text-muted small mt-2"),

            dcc.Loading(
                id="loading-output",
                type="default",
                children=dbc.Card(
                    dbc.CardBody([
                        html.Div(id="output-prediction", className="fs-4 text-center fw-bold py-2"),
                        html.Hr(),
                        html.H4("Fatores que Influenciam a Previsão", className="text-center mb-3"),
                        html.P(
                            "Este gráfico mostra as características que mais contribuíram para a previsão de hipertensão para ESTE PACIENTE ESPECÍFICO. As barras azuis indicam fatores que diminuem o risco de hipertensão, enquanto as barras vermelhas indicam fatores que o aumentam. Uma barra mais longa significa maior impacto.",
                            className="text-muted text-center small"),
                        dcc.Graph(id="shap-explanation-graph", config={'displayModeBar': False})
                    ]),
                    id="prediction-card",
                    className="mt-4 shadow-sm border-0 d-none"
                )
            ),
            # --- SEÇÃO DE RECOMENDAÇÕES ---
            dbc.Card(
                dbc.CardBody([
                    html.H4([
                        html.I(className="bi bi-lightbulb-fill me-2 text-warning"),
                        "Recomendações de Prevenção"
                    ], className="card-title text-center mb-3 text-primary"),
                    html.P(
                        "Aqui estão sugestões personalizadas para manter ou melhorar sua saúde cardiovascular, geradas por inteligência artificial.",
                        className="text-muted text-center small mb-4"
                    ),
                    html.Div(id="output-recommendations", className="text-left small"),
                    html.P(
                        "Estas recomendações servem como um guia informativo e não substituem o diagnóstico ou o aconselhamento médico profissional.",
                        className="text-muted text-center small mt-4 pt-3 border-top")
                ]),
                id="recommendations-card",
                className="mt-4 shadow-sm border-0 d-none"
            ),
            # --- NOVA SEÇÃO: HISTÓRICO DE PREVISÕES ---
            dbc.Card(
                dbc.CardBody([
                    html.H4([
                        html.I(className="bi bi-clock-history me-2 text-info"),
                        "Meu Histórico de Previsões"
                    ], className="card-title text-center mb-3 text-primary"),
                    html.P(
                        "Visualize suas simulações anteriores e os resultados da previsão de hipertensão.",
                        className="text-muted text-center small mb-4"
                    ),
                    html.Div(id="user-predictions-table-container"),  # Contêiner para a tabela
                    html.P("O histórico é atualizado automaticamente após cada nova simulação.",
                           className="text-muted text-center small mt-4 pt-3 border-top")
                ]),
                id="history-card",
                className="mt-4 shadow-sm border-0"
                # Visível por padrão, ou d-none se quiser ocultar até a primeira simulação
            )
            # --- FIM DA NOVA SEÇÃO ---
        ]),
        className="mb-5 shadow-lg border-primary"
    ),

    html.Footer(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.P("Desenvolvido com Dash e Machine Learning. © 2025 - Rafael Nascimento/Analista de Dados",
                           className="text-muted text-center py-3 mb-0"),
                    width=12
                )
            ])
        ]),
        className="bg-light mt-5"
    )
], fluid=True, className="py-4 bg-light")


# === 6. Callback para Previsão e Interpretabilidade ===
@app.callback(
    [Output("output-prediction", "children"),
     Output("prediction-card", "className"),
     Output("error-age", "children"),
     Output("error-sal", "children"),
     Output("error-estresse", "children"),
     Output("error-sono", "children"),
     Output("error-bmi", "children"),
     Output("error-bp", "children"),
     Output("error-med", "children"),
     Output("error-fam", "children"),
     Output("error-ex", "children"),
     Output("error-smk", "children"),
     Output("shap-explanation-graph", "figure"),
     Output("db-save-status", "children"),

     # Saídas para atualizar as estatísticas do dataset
     Output("total-original-patients", "children"),
     Output("original-hypertensive-patients", "children"),
     Output("original-non-hypertensive-patients", "children"),
     Output("total-simulated-patients", "children"),
     Output("simulated-hypertensive-patients", "children"),
     Output("simulated-non-hypertensive-patients", "children"),
     Output("total-combined-patients", "children"),
     Output("combined-hypertensive-patients", "children"),
     Output("combined-non-hypertensive-patients", "children"),

     # Saídas para atualizar os gráficos
     Output("graph-age-distribution", "figure"),
     Output("graph-bmi-distribution", "figure"),
     Output("graph-age-hypertension", "figure"),
     Output("graph-salt-hypertension", "figure"),
     Output("graph-bp-history-hypertension", "figure"),
     Output("graph-smoking-hypertension", "figure"),
     Output("graph-correlation-heatmap", "figure"),
     Output("graph-stress-hypertension", "figure"),
     Output("graph-scatter-matrix", "figure"),
     Output("output-recommendations", "children"),
     Output("recommendations-card", "className"),
     Output("user-predictions-table-container", "children")  # NOVA SAÍDA para a tabela de histórico
     ],
    Input("btn-predict", "n_clicks"),
    State("input-age", "value"),
    State("input-sal", "value"),
    State("input-estresse", "value"),
    State("input-sono", "value"),
    State("input-bmi", "value"),
    State("input-bp", "value"),
    State("input-med", "value"),
    State("input-fam", "value"),
    State("input-ex", "value"),
    State("input-smk", "value")
)
def prever(n_clicks, age, sal, stress, sono, bmi, bp, med, fam, ex, smk):
    initial_error_messages = [""] * 10
    prediction_card_class = "mt-4 shadow-sm border-0 d-none"
    recommendations_card_class = "mt-4 shadow-sm border-0 d-none"
    empty_shap_figure = {}
    db_status_message = ""
    empty_recommendations = ""
    empty_table = html.Div()  # Componente vazio para a tabela

    df_original_current, df_simulated_current, df_for_graphs_current = load_and_process_data()
    current_stats_output = get_dashboard_stats(df_original_current, df_simulated_current, df_for_graphs_current)
    current_graphs_output = generate_all_graphs(df_for_graphs_current)

    # --- LER E EXIBIR A TABELA DE HISTÓRICO SEMPRE ---
    try:
        df_simulated_history = pd.read_csv(SIMULATED_PATIENTS_DB)
        # Formatar o Timestamp para uma leitura mais amigável
        if 'Timestamp' in df_simulated_history.columns:
            df_simulated_history['Timestamp'] = pd.to_datetime(df_simulated_history['Timestamp']).dt.strftime(
                '%d/%m/%Y %H:%M')

        # Reverter a ordem para mostrar as mais recentes primeiro, e limitar a, por exemplo, 10 registros
        df_simulated_history_display = df_simulated_history.tail(10).sort_values(by='Timestamp', ascending=False)

        # Selecionar colunas para exibição e renomear para melhor apresentação
        cols_to_display = [
            "Timestamp", "Age", "BMI", "Salt_Intake", "Stress_Score",
            "Predicted_Hypertension", "Prediction_Probability"
        ]
        display_names = {
            "Timestamp": "Data/Hora", "Age": "Idade", "BMI": "IMC",
            "Salt_Intake": "Sal (g/dia)", "Stress_Score": "Estresse",
            "Predicted_Hypertension": "Previsão", "Prediction_Probability": "Probabilidade (%)"
        }

        # Certificar-se de que todas as colunas existem antes de tentar selecioná-las
        cols_to_display_existing = [col for col in cols_to_display if col in df_simulated_history_display.columns]
        df_for_table = df_simulated_history_display[cols_to_display_existing].rename(columns=display_names)

        user_predictions_table = html.Div([
            dbc.Table.from_dataframe(df_for_table, striped=True, bordered=True, hover=True,
                                     className="mt-3 text-center"),
            html.P("Nenhuma previsão registrada ainda.",
                   className="text-muted text-center mt-3") if df_for_table.empty else html.Div()
        ])
    except pd.errors.EmptyDataError:
        user_predictions_table = html.P("Nenhuma previsão registrada ainda.", className="text-muted text-center mt-3")
    except FileNotFoundError:
        user_predictions_table = html.P("Arquivo de histórico não encontrado.",
                                        className="text-danger text-center mt-3")
    except Exception as e:
        user_predictions_table = html.P(f"Erro ao carregar histórico: {str(e)}",
                                        className="text-danger text-center mt-3")

    if not n_clicks:
        return "", prediction_card_class, \
            *initial_error_messages, \
            empty_shap_figure, \
            db_status_message, \
            *current_stats_output, \
            *current_graphs_output, \
            empty_recommendations, recommendations_card_class, \
            user_predictions_table  # Retorna a tabela vazia ou com histórico existente

    # Verifica se o modelo foi carregado com sucesso antes de tentar fazer previsões
    if MODEL_PIPELINE is None or PREPROCESSOR is None or CLASSIFIER is None:
        db_status_message_on_error = "❌ Erro: Modelo ou componentes não carregados. Previsão indisponível."
        return "❌ Previsão indisponível.", \
            "mt-4 fs-5 text-danger", \
            *initial_error_messages, empty_shap_figure, db_status_message_on_error, \
            *current_stats_output, *current_graphs_output, \
            empty_recommendations, recommendations_card_class, \
            user_predictions_table

    try:
        error_msgs = initial_error_messages[:]
        has_errors = False

        inputs_raw = {
            "Age": age, "Salt_Intake": sal, "Stress_Score": stress,
            "Sleep_Duration": sono, "BMI": bmi, "BP_History": bp,
            "Medication": med, "Family_History": fam, "Exercise_Level": ex,
            "Smoking_Status": smk
        }

        input_keys_for_errors = list(inputs_raw.keys())

        for i, (key, value) in enumerate(inputs_raw.items()):
            if value is None:
                error_msgs[i] = "Este campo é obrigatório."
                has_errors = True

        if age is not None and not (1 <= age <= 120):
            error_msgs[input_keys_for_errors.index("Age")] = "Idade deve ser entre 1 e 120."
            has_errors = True
        if sal is not None and not (0 <= sal <= 20):
            error_msgs[input_keys_for_errors.index("Salt_Intake")] = "Sal deve ser entre 0 e 20g/dia."
            has_errors = True
        if stress is not None and not (0 <= stress <= 10):
            error_msgs[input_keys_for_errors.index("Stress_Score")] = "Estresse deve ser entre 0 e 10."
            has_errors = True
        if sono is not None and not (0 <= sono <= 24):
            error_msgs[input_keys_for_errors.index("Sleep_Duration")] = "Sono deve ser entre 0 e 24h."
            has_errors = True
        if bmi is not None and not (10 <= bmi <= 80):
            error_msgs[input_keys_for_errors.index("BMI")] = "IMC deve ser entre 10 e 80."
            has_errors = True

        if has_errors:
            return "⚠️ Por favor, corrija os erros nos campos marcados.", \
                prediction_card_class, \
                *error_msgs, \
                empty_shap_figure, \
                db_status_message, \
                *current_stats_output, \
                *current_graphs_output, \
                empty_recommendations, recommendations_card_class, \
                user_predictions_table  # Retorna a tabela mesmo com erros

        input_data_for_df = {
            "Age": int(age),
            "Salt_Intake": float(sal),
            "Stress_Score": int(stress),
            "Sleep_Duration": float(sono),
            "BMI": float(bmi),
            "BP_History": str(bp),
            "Medication": str(med),
            "Family_History": str(fam),
            "Exercise_Level": str(ex),
            "Smoking_Status": str(smk)
        }

        input_df = pd.DataFrame([input_data_for_df], columns=ORIGINAL_MODEL_FEATURES)

        pred = MODEL_PIPELINE.predict(input_df)[0]
        prob = MODEL_PIPELINE.predict_proba(input_df)[0][1]

        resultado = "Hipertenso" if pred == 1 else "Não Hipertenso"

        X_transformed_for_shap = PREPROCESSOR.transform(input_df)

        explainer = shap.TreeExplainer(CLASSIFIER, SHAP_BACKGROUND_DATA)
        shap_values = explainer.shap_values(X_transformed_for_shap)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_for_class_1 = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_for_class_1 = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_values_for_class_1 = shap_values[0]
        else:
            raise ValueError(
                f"Formato inesperado de shap_values: {type(shap_values)}, dim: {getattr(shap_values, 'ndim', 'N/A')}")

        if not isinstance(shap_values_for_class_1, np.ndarray) or shap_values_for_class_1.ndim != 1 or len(
                shap_values_for_class_1) != len(TRANSFORMED_COLUMN_NAMES):
            raise ValueError(f"shap_values_for_class_1 não é 1D ou tem comprimento incorreto após processamento. "
                             f"Shape final: {getattr(shap_values_for_class_1, 'shape', 'N/A')}, "
                             f"Comprimento final: {len(shap_values_for_class_1) if hasattr(shap_values_for_class_1, '__len__') else 'N/A'}. "
                             f"Comprimento esperado: {len(TRANSFORMED_COLUMN_NAMES)}."
                             f"\nDEBUG INFO: Tipo original shap_values: {type(shap_values)}, "
                             f"Shape original shap_values: {getattr(shap_values, 'shape', 'N/A')}.")

        shap_df = pd.DataFrame({
            'feature': TRANSFORMED_COLUMN_NAMES,
            'shap_value': shap_values_for_class_1
        })

        shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values(by='abs_shap_value', ascending=False).head(5)

        shap_figure = px.bar(
            shap_df,
            x='shap_value',
            y='feature',
            orientation='h',
            title='Fatores Mais Influentes na Previsão (SHAP Values)',
            labels={'shap_value': 'Impacto SHAP', 'feature': 'Característica'},
            color='shap_value',
            color_continuous_scale=px.colors.sequential.RdBu,
            template="plotly_white"
        )
        shap_figure.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})

        if pred == 1:
            prediction_class = "mt-4 shadow-lg border-danger"
            main_prediction_message = html.Span([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                f"Previsão: {resultado} | Probabilidade: {round(prob * 100, 2)}%"
            ], className="text-danger")
            predicted_hypertension_text = "Hipertenso"
        else:
            prediction_class = "mt-4 shadow-lg border-success"
            main_prediction_message = html.Span([
                html.I(className="bi bi-check-circle-fill me-2"),
                f"Previsão: {resultado} | Probabilidade: {round(prob * 100, 2)}%"
            ], className="text-success")
            predicted_hypertension_text = "Não Hipertenso"

        # === SALVAR DADOS DA SIMULAÇÃO NO CSV ===
        try:
            current_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            simulated_data = inputs_raw.copy()
            simulated_data["Predicted_Hypertension"] = predicted_hypertension_text
            simulated_data["Prediction_Probability"] = round(prob * 100, 2)
            simulated_data["Timestamp"] = current_timestamp
            # Não há Patient_ID no SIMULATED_PATIENTS_COLS original, então não tente salvar aqui.
            # Se você deseja Patient_ID, precisaremos de um formulário para o analista coletá-lo
            # e modificar SIMULATED_PATIENTS_COLS para incluir 'Patient_ID'.

            df_to_save = pd.DataFrame([simulated_data], columns=SIMULATED_PATIENTS_COLS)
            # Verifica se o arquivo existe e se ele está vazio para saber se deve escrever o cabeçalho
            file_exists = os.path.exists(SIMULATED_PATIENTS_DB)
            if file_exists and os.path.getsize(SIMULATED_PATIENTS_DB) == 0:
                df_to_save.to_csv(SIMULATED_PATIENTS_DB, mode='a', header=True, index=False)
            else:
                df_to_save.to_csv(SIMULATED_PATIENTS_DB, mode='a', header=False, index=False)
            db_status_message = "✅ Dados da simulação salvos com sucesso!"
        except Exception as db_e:
            db_status_message = f"❌ Erro ao salvar dados da simulação: {str(db_e)}"
            print(f"ERRO AO SALVAR NO CSV: {db_e}")

        # Recarregar dados E A TABELA DE HISTÓRICO APÓS SALVAR
        df_original_updated, df_simulated_updated, df_for_graphs_updated = load_and_process_data()
        updated_stats_output = get_dashboard_stats(df_original_updated, df_simulated_updated, df_for_graphs_updated)
        updated_graphs_output = generate_all_graphs(df_for_graphs_updated)

        # Atualizar a tabela de histórico aqui também
        try:
            df_simulated_history = pd.read_csv(SIMULATED_PATIENTS_DB)
            if 'Timestamp' in df_simulated_history.columns:
                df_simulated_history['Timestamp'] = pd.to_datetime(df_simulated_history['Timestamp']).dt.strftime(
                    '%d/%m/%Y %H:%M')
            df_simulated_history_display = df_simulated_history.tail(10).sort_values(by='Timestamp', ascending=False)
            cols_to_display_existing = [col for col in cols_to_display if col in df_simulated_history_display.columns]
            df_for_table = df_simulated_history_display[cols_to_display_existing].rename(columns=display_names)
            user_predictions_table_updated = html.Div([
                dbc.Table.from_dataframe(df_for_table, striped=True, bordered=True, hover=True,
                                         className="mt-3 text-center"),
                html.P("Nenhuma previsão registrada ainda.",
                       className="text-muted text-center mt-3") if df_for_table.empty else html.Div()
            ])
        except pd.errors.EmptyDataError:
            user_predictions_table_updated = html.P("Nenhuma previsão registrada ainda.",
                                                    className="text-muted text-center mt-3")
        except FileNotFoundError:
            user_predictions_table_updated = html.P("Arquivo de histórico não encontrado.",
                                                    className="text-danger text-center mt-3")
        except Exception as e:
            user_predictions_table_updated = html.P(f"Erro ao carregar histórico: {str(e)}",
                                                    className="text-danger text-center mt-3")

        generated_recommendations = generate_prevention_recommendations(inputs_raw, resultado)
        recommendations_card_class = "mt-4 shadow-sm border-0"

        return main_prediction_message, \
            prediction_class, \
            *error_msgs, \
            shap_figure, \
            db_status_message, \
            *updated_stats_output, \
            *updated_graphs_output, \
            generated_recommendations, recommendations_card_class, \
            user_predictions_table_updated  # Retorna a tabela atualizada

    except Exception as e:
        print(f"ERRO INTERNO DETALHADO NO CALLBACK: {e}")
        db_status_message_on_error = f"❌ Erro interno ao processar previsão. Por favor, tente novamente. Detalhes: {str(e)[:100]}..."
        return f"❌ Erro interno ao processar previsão. Por favor, tente novamente.", \
            "mt-4 fs-5 text-danger", \
            *initial_error_messages, empty_shap_figure, db_status_message_on_error, \
            *current_stats_output, *current_graphs_output, \
            empty_recommendations, recommendations_card_class, \
            user_predictions_table  # Retorna a tabela mesmo com erro (sem atualizar)


# --- ROTAS FLASK ADICIONAIS ---
@server.route('/')
def index():
    return render_template_string(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plataforma de Saúde | Bem-vindo</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    font-family: 'Montserrat', sans-serif;
                    background: linear-gradient(135deg, #e0f2f7 0%, #bbdefb 100%); /* Gradiente suave */
                    color: #343a40;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    overflow: hidden;
                    text-align: center;
                }}
                .main-container {{
                    background-color: rgba(255, 255, 255, 0.95); /* Fundo branco semitransparente */
                    border-radius: 20px; /* Bordas mais arredondadas */
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2); /* Sombra mais pronunciada */
                    padding: 60px;
                    max-width: 900px;
                    width: 90%;
                    position: relative;
                    z-index: 10;
                    animation: fadeIn 1s ease-out; /* Animação de fade-in */
                }}
                h1 {{
                    font-family: 'Montserrat', sans-serif;
                    font-weight: 700;
                    color: #007bff; /* Azul vibrante */
                    margin-bottom: 25px;
                    font-size: 3.5rem; /* Tamanho maior */
                    text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
                }}
                p.lead {{
                    color: #555;
                    line-height: 1.8;
                    margin-bottom: 40px;
                    font-size: 1.25rem;
                    font-weight: 400;
                }}
                .btn-access {{
                    background-color: #28a745; /* Verde saúde */
                    border-color: #28a745;
                    font-weight: 600;
                    padding: 15px 40px; /* Mais preenchimento */
                    border-radius: 35px; /* Mais arredondado */
                    font-size: 1.3rem; /* Texto maior no botão */
                    transition: all 0.3s ease;
                    text-transform: uppercase; /* Texto em maiúsculas */
                    letter-spacing: 1px; /* Espaçamento entre letras */
                }}
                .btn-access:hover {{
                    background-color: #218838;
                    border-color: #1e7e34;
                    transform: translateY(-3px); /* Efeito de elevação */
                    box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
                }}
                .bi {{
                    vertical-align: middle;
                    margin-right: 12px; /* Mais espaço para o ícone */
                    font-size: 1.8rem; /* Ícone maior */
                }}
                .icon-heart {{
                    font-size: 4rem; /* Ícone de coração ainda maior */
                    color: #dc3545; /* Vermelho vibrante */
                    margin-bottom: 25px;
                    animation: pulse 2s infinite; /* Animação de pulsação */
                }}

                /* Animações CSS */
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                    100% {{ transform: scale(1); }}
                }}

                /* Efeitos de fundo de partículas/formas geométricas */
                .shape {{
                    position: absolute;
                    background-color: rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    animation: float 20s infinite ease-in-out;
                    z-index: 1;
                }}
                .shape.one {{ top: 10%; left: 5%; width: 100px; height: 100px; animation-delay: 0s; }}
                .shape.two {{ bottom: 15%; right: 10%; width: 120px; height: 120px; animation-delay: 3s; border-radius: 30%; }}
                .shape.three {{ top: 20%; right: 20%; width: 80px; height: 80px; animation-delay: 7s; }}
                .shape.four {{ bottom: 5%; left: 20%; width: 150px; height: 150px; animation-delay: 10s; }}

                @keyframes float {{
                    0% {{ transform: translateY(0) rotate(0deg); opacity: 0.5; }}
                    25% {{ transform: translateY(-20px) rotate(45deg); opacity: 0.7; }}
                    50% {{ transform: translateY(0) rotate(90deg); opacity: 0.5; }}
                    75% {{ transform: translateY(20px) rotate(135deg); opacity: 0.7; }}
                    100% {{ transform: translateY(0) rotate(180deg); opacity: 0.5; }}
                }}
            </style>
        </head>
        <body>
            <div class="shape one"></div>
            <div class="shape two"></div>
            <div class="shape three"></div>
            <div class="shape four"></div>

            <div class="main-container">
                <i class="bi bi-heart-fill icon-heart"></i>
                <h1>Sua Saúde, Nossa Prioridade</h1>
                <p class="lead">
                    Acompanhe seu risco de hipertensão com insights baseados em Machine Learning.
                    Decisões informadas para uma vida mais longa e saudável.
                </p>
                <a href="{DASH_APP_ROUTE_PREFIX}" class="btn btn-access">
                    <i class="bi bi-graph-up"></i> Acessar Dashboard de Saúde
                </a>
            </div>
        </body>
        </html>
    """)


# --- INICIALIZAÇÃO DO SERVIDOR ---
# if __name__ == "__main__":
    # server.run(debug=True, port=8050)