import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.gera_sample import gerar_dataframe_amostra
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Streamlit Web Application
st.title("Financial Market Fraud Detection System")

# Step 1: Data Collection & Preparation
data = gerar_dataframe_amostra(1000)

data["timestamp"] = pd.to_datetime(data["timestamp"])
st.write("Dataset Preview:")
st.dataframe(data.head(20))

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
degree_freq = dict(nx.degree(G))

# Step 3: Feature Engineering
features = []
for node in G.nodes:
    centrality_value = centrality[node]
    clustering_coef = clustering[node]
    transaction_count = len(list(G.edges(node)))
    degree = degree_freq[node]
    features.append(
        [node, centrality_value, clustering_coef, transaction_count, degree]
    )

features_df = pd.DataFrame(
    features,
    columns=["node", "centrality", "clustering_coef", "transaction_count", "degree"],
)

# Step 4: Unsupervised Learning - Clustering for Anomaly Detection
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df.drop(columns=["node"]))

kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_features)
features_df["cluster"] = kmeans.labels_

st.write("Network Metrics Preview:")
st.dataframe(features_df.sort_values(by=["clustering_coef", "degree"], ascending=False))

# Plot Clusters
st.subheader("Cluster Visualization")
fig = px.scatter(
    features_df,
    x="clustering_coef",
    y="transaction_count",
    color="cluster",
    title="Clusters of Financial Transactions",
    labels={
        "clustering_coef": "Clustering Coefficient",
        "transaction_count": "Transaction Count",
    },
)
st.plotly_chart(fig)

st.subheader("Cluster Assignment Results")
st.write("Cluster Labels:")
st.dataframe(features_df["cluster"].value_counts())

# Step 4: Degree Frequency Analysis
st.subheader("Degree Frequency Analysis")
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_count = pd.Series(degree_sequence).value_counts().sort_index()

fig = px.bar(
    x=degree_count.index,
    y=degree_count.values,
    labels={"x": "Degree", "y": "Frequency"},
)
fig.update_layout(
    title="Degree Frequency Analysis",
    xaxis_title="Degree",
    yaxis_title="Frequency",
    xaxis=dict(tickmode="linear", tick0=1),
)
st.plotly_chart(fig)

# Step 6: Visualization & Analysis
st.subheader("Transaction Network Visualization")
pos = nx.spring_layout(G, k=0.15)
edges = G.edges(data=True)
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines",
)

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    hoverinfo="text",
    marker=dict(showscale=True, colorscale="YlGnBu", size=10, color=[], line_width=2),
)

node_adjacencies = []
node_text = []
for node in G.nodes():
    node_adjacencies.append(len(list(G.adj[node])))
    node_text.append(f"Node {node}, Degree: {len(list(G.adj[node]))}")

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="Financial Transaction Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

st.plotly_chart(fig)
