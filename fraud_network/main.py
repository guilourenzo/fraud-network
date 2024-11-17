import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.gera_sample import gerar_dataframe_amostra
import matplotlib.pyplot as plt
import shap
import streamlit as st


# Streamlit Web Application
st.title("Financial Market Fraud Detection System")

# Step 1: Data Collection & Preparation
uploaded_file = st.file_uploader(
    "Upload your financial transactions dataset (CSV format)", type="csv"
)

# Exemplo de uso:
num_rows = 100
sender_pool = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
receiver_pool = sender_pool.copy()
amount_range = (100, 5000)
date_range = '2023-01-01'


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.write("Using sample dataset as default.")
    data = gerar_dataframe_amostra(num_rows, sender_pool, receiver_pool, amount_range, date_range)

data["timestamp"] = pd.to_datetime(data["timestamp"])
st.write("Dataset Preview:")
st.dataframe(data.head())

# Step 2: Network Modeling
G = nx.from_pandas_edgelist(
    data,
    source="sender",
    target="receiver",
    edge_attr="amount",
    create_using=nx.DiGraph(),
)

# Display Network Metrics
st.subheader("Network Metrics")
centrality = nx.degree_centrality(G)
clustering = nx.clustering(G.to_undirected())
metrics_df = pd.DataFrame(
    {
        "Node": list(centrality.keys()),
        "Centrality": list(centrality.values()),
        "Clustering Coefficient": [clustering[node] for node in centrality.keys()],
    }
)
st.write("Network Metrics Preview:")
st.dataframe(metrics_df.head())

# Step 3: Feature Engineering
features = []
for node in G.nodes:
    centrality_value = centrality[node]
    clustering_coef = clustering[node]
    transaction_count = len(list(G.edges(node)))
    features.append([node, centrality_value, clustering_coef, transaction_count])

features_df = pd.DataFrame(
    features, columns=["node", "centrality", "clustering_coef", "transaction_count"]
)
st.write("Features DF")
st.dataframe(features_df)

# Step 4: Unsupervised Learning - Clustering for Anomaly Detection
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df.drop(columns=["node"]))

clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled_features)
features_df["anomaly"] = clustering.labels_

st.subheader("Anomaly Detection Results")
st.write("Anomaly Labels:")
st.dataframe(features_df["anomaly"].value_counts())

# Step 5: Fraud Detection - Supervised Learning
# Assuming we have labeled data for training
if "fraud_label" in data.columns:
    labeled_data = data.copy()
    labeled_data = pd.merge(labeled_data, features_df, how='left', left_on='sender', right_on='node')
    X = labeled_data[["centrality", "clustering_coef", "transaction_count"]]
    y = labeled_data["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    st.subheader("Fraud Detection - Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Interpret the Model Using SHAP
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test)

    st.subheader("SHAP Summary Plot")
    # Criação explícita de uma figura
    fig, ax = plt.subplots()

    # Gerar o gráfico do SHAP na figura criada
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.gcf().set_size_inches(fig.get_size_inches())  # Garantir que o tamanho se ajuste ao da figura criada

    # Renderizar o gráfico no Streamlit
    st.pyplot(fig, bbox_inches="tight")

# Step 6: Visualization & Analysis
st.subheader("Transaction Network Visualization")

# Criação explícita de uma figura
fig, ax = plt.subplots(figsize=(10, 10))

# Configuração do layout e desenho do grafo
pos = nx.spring_layout(G, k=0.15)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color="blue", ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

# Título do gráfico
ax.set_title("Financial Transaction Network")

# Renderizar o gráfico com Streamlit
st.pyplot(fig)