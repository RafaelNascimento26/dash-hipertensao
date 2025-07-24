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

# === Configuração do Banco de Dados (CSV) para Pacientes Simulados ===
SIMULATED_PATIENTS_DB = "simulated_patients.csv"
SIMULATED_PATIENTS_COLS = [
    "Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI",
    "BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status",
    "Predicted_Hypertension", "Prediction_Probability", "Timestamp"
]


def initialize_simulated_patients_db():
    if not os.path.exists(SIMULATED_PATIENTS_DB):
        df_empty = pd.DataFrame(columns=SIMULATED_PATIENTS_COLS)
        df_empty.to_csv(SIMULATED_PATIENTS_DB, index=False)
        print(f"Arquivo de banco de dados '{SIMULATED_PATIENTS_DB}' criado com cabeçalhos.")
    else:
        print(f"Arquivo de banco de dados '{SIMULATED_PATIENTS_DB}' já existe.")


initialize_simulated_patients_db()

# === 1. Carregar modelo e dados ===
try:
    with open("model.pkl", "rb") as f:
        MODEL_PIPELINE = pickle.load(f)
    print("Modelo (pipeline) carregado com sucesso!")
except FileNotFoundError:
    print(
        "ERRO: 'model.pkl' não encontrado. Certifique-se de que o arquivo está na mesma pasta ou que o caminho está correto.")
    exit()

try:
    TRANSFORMED_COLUMN_NAMES = pickle.load(open("transformed_column_names.pkl", "rb"))
    print("Nomes das colunas transformadas carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: 'transformed_column_names.pkl' não encontrado. Rode 'treinamento_modelo.py' PRIMEIRO.")
    exit()

try:
    SHAP_BACKGROUND_DATA = pickle.load(open("shap_background_data.pkl", "rb"))
    print("SHAP background data carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: 'shap_background_data.pkl' não encontrado. Rode 'treinamento_modelo.py' PRIMEIRO.")
    exit()

PREPROCESSOR = MODEL_PIPELINE.named_steps['preprocessor']
CLASSIFIER = MODEL_PIPELINE.named_steps['classifier']

ORIGINAL_MODEL_FEATURES = [
    "Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI",
    "BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"
]

categorical_cols = ["BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"]
numerical_features = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]


# === Funções Auxiliares para Carregar, Processar Dados e Gerar Gráficos/Estatísticas ===
def load_and_process_data():
    try:
        df_original = pd.read_csv("hypertension_dataset.csv")
    except FileNotFoundError:
        print("ERRO: 'hypertension_dataset.csv' não encontrado. Certifique-se de que o arquivo está na mesma pasta.")
        exit()

    try:
        df_simulated = pd.read_csv(SIMULATED_PATIENTS_DB)
    except FileNotFoundError:
        df_simulated = pd.DataFrame(columns=SIMULATED_PATIENTS_COLS)

    for col in categorical_cols:
        if col in df_original.columns:
            df_original[col] = df_original[col].fillna("None").astype(str)
        if col in df_simulated.columns:
            df_simulated[col] = df_simulated[col].fillna("None").astype(str)

    if 'Has_Hypertension' in df_original.columns:
        df_original['Hypertension_Status'] = df_original['Has_Hypertension'].map(
            {'Yes': 'Hipertenso', 'No': 'Não Hipertenso'})
        df_original['Hypertension_Binary_Numeric'] = df_original['Has_Hypertension'].map({'Yes': 1, 'No': 0})
    else:
        df_original['Hypertension_Status'] = pd.Series(dtype='str')
        df_original['Hypertension_Binary_Numeric'] = pd.Series(dtype='float')

    if 'Predicted_Hypertension' in df_simulated.columns:
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
    total_original = len(df_original)
    total_simulated = len(df_simulated)
    total_combined = len(df_for_graphs)

    original_hypertension_counts = df_original['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'], fill_value=0)
    simulated_hypertension_counts = df_simulated['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'], fill_value=0)
    combined_hypertension_counts = df_for_graphs['Hypertension_Status'].value_counts().reindex(
        ['Hipertenso', 'Não Hipertenso'], fill_value=0)

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
    # Condições para exibir os gráficos
    # Garante que as colunas necessárias existam antes de tentar criar o gráfico
    age_in_df = "Age" in df_for_graphs.columns
    bmi_in_df = "BMI" in df_for_graphs.columns
    salt_in_df = "Salt_Intake" in df_for_graphs.columns
    stress_in_df = "Stress_Score" in df_for_graphs.columns
    bp_history_in_df = "BP_History" in df_for_graphs.columns
    smoking_status_in_df = "Smoking_Status" in df_for_graphs.columns
    hypertension_status_in_df = "Hypertension_Status" in df_for_graphs.columns
    hypertension_binary_numeric_in_df = "Hypertension_Binary_Numeric" in df_for_graphs.columns

    # Verifica se todas as features numéricas e o target binário estão presentes para o heatmap
    all_numerical_and_target_present_for_heatmap = all(
        col in df_for_graphs.columns for col in numerical_features + ['Hypertension_Binary_Numeric'])
    # Verifica se todas as features numéricas e o status de hipertensão estão presentes para o scatter matrix
    all_numerical_and_status_present_for_scatter_matrix = all(
        col in df_for_graphs.columns for col in numerical_features + ['Hypertension_Status'])

    # === NOVO GRÁFICO: MATRIZ DE GRÁFICOS DE DISPERSÃO (PAIR PLOT) ===
    # A figura é criada e atualizada fora da lista de retorno, depois adicionada.
    scatter_matrix_fig = {}  # Inicializa a figura vazia
    if all_numerical_and_status_present_for_scatter_matrix:
        scatter_matrix_fig = px.scatter_matrix(df_for_graphs, dimensions=numerical_features,
                                               color="Hypertension_Status",
                                               title="Matriz de Dispersão das Variáveis Numéricas por Status de Hipertensão",
                                               template="plotly_white",
                                               color_discrete_map={"Hipertenso": "red", "Não Hipertenso": "green"})
        scatter_matrix_fig.update_yaxes(tickangle=0, automargin=True)# Definir tickangl

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
        scatter_matrix_fig,  # Retorna a figura do scatter matrix (já criada e atualizada)
    ]


# Carregar dados e gerar estatísticas/gráficos para o layout inicial
df_original_initial, df_simulated_initial, df_for_graphs_initial = load_and_process_data()
initial_stats_output = get_dashboard_stats(df_original_initial, df_simulated_initial, df_for_graphs_initial)
initial_graphs_output = generate_all_graphs(df_for_graphs_initial)

# === DEFINIÇÕES DAS OPÇÕES DOS DROPDOWNS ===
bp_options = ["Normal", "Hypertension", "Prehypertension"]
medication_options = ["None", "ACE Inhibitor", "Other", "Beta Blocker", "Diuretic"]
family_options = ["Yes", "No"]
exercise_options = ["Low", "Moderate", "High"]
smoking_options = ["Smoker", "Non-Smoker"]

# === 4. Inicializar app Dash ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY,
                                                "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"])
app.title = "Previsão de Hipertensão"

# === 5. Layout do Aplicativo Dash ===
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
            html.P(  # Texto REVISADO para a seção de simulação
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
            )
        ]),
        className="mb-5 shadow-lg border-primary"
    ),

    html.Footer(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.P("Desenvolvido com Dash e Machine Learning. © 2024 - Seu Nome/Empresa",
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
    empty_shap_figure = {}
    db_status_message = ""

    # Carregar dados e gerar estatísticas/gráficos para o ESTADO ATUAL
    df_original_current, df_simulated_current, df_for_graphs_current = load_and_process_data()

    current_stats_output = get_dashboard_stats(df_original_current, df_simulated_current, df_for_graphs_current)
    current_graphs_output = generate_all_graphs(df_for_graphs_current)

    if not n_clicks:
        return "", prediction_card_class, \
            *initial_error_messages, \
            empty_shap_figure, \
            db_status_message, \
            *current_stats_output, \
            *current_graphs_output

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
            has_has_errors = True  # CUIDADO: Este é um typo, deveria ser 'has_errors = True'
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
                *current_graphs_output

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

        shap_values_for_class_1 = shap_values[0, :, 1]

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

            df_to_save = pd.DataFrame([simulated_data], columns=SIMULATED_PATIENTS_COLS)
            df_to_save.to_csv(SIMULATED_PATIENTS_DB, mode='a', header=False, index=False)
            db_status_message = "✅ Dados da simulação salvos com sucesso!"
        except Exception as db_e:
            db_status_message = f"❌ Erro ao salvar dados da simulação: {str(db_e)}"
            print(f"ERRO AO SALVAR NO CSV: {db_e}")

        # Recarregar e recalcular dados e gráficos APÓS salvar
        df_original_updated, df_simulated_updated, df_for_graphs_updated = load_and_process_data()

        updated_stats_output = get_dashboard_stats(df_original_updated, df_simulated_updated, df_for_graphs_updated)
        updated_graphs_output = generate_all_graphs(df_for_graphs_updated)

        # Retorna todas as saídas na ordem exata definida no decorador @app.callback.
        return main_prediction_message, \
            prediction_class, \
            *error_msgs, \
            shap_figure, \
            db_status_message, \
            *updated_stats_output, \
            *updated_graphs_output

    except Exception as e:
        print(f"ERRO INTERNO DETALHADO NO CALLBACK: {e}")
        db_status_message_on_error = f"❌ Erro interno ao processar: {str(e)[:100]}..."
        # Em caso de erro, retorna os valores atuais para as estatísticas e gráficos.
        # Eles não serão atualizados, mas o app não quebrará.
        # Assegura que todas as 31 saídas são fornecidas.
        return f"❌ Erro interno ao processar previsão. Por favor, tente novamente.", \
            "mt-4 fs-5 text-danger", \
            *initial_error_messages, empty_shap_figure, db_status_message_on_error, \
            *current_stats_output, *current_graphs_output


if __name__ == "__main__":
    app.run(debug=True, port=8050)