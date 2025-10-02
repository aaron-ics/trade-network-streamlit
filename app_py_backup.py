from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import numpy as np
from st_cytoscape import cytoscape

from utils import prettify_metric_name
from src.invoices.invoice_simulator import make_invoice_data, calculate_company_metrics


def wide_space_default():
    st.set_page_config(
        page_title="Trade Network Dashboard",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )


wide_space_default()

# Add custom CSS styling with the new color palette
st.markdown("""
<style>
    /* Color variables as per your reference */
    :root {
        /* Color Palette */
        --color-primary: #008040;        /* Deep Emerald Green (Accent/Current) */
        --color-background: #FFFFFF;     /* Pure White */
        --color-text-dark: #111111;      /* Near Black (Headings, Metrics) */
        --color-text-light: #666666;     /* Medium Gray (Subtle text, inactive links) */
        --color-border: #EEEEEE;         /* Very light gray (Separators, card borders) */
        --color-secondary-bar: #DDDDDD;  /* Light gray (Applied/Secondary bars) */
        --color-limit: #FF0000;          /* Red (Limit line) */
        
        /* Typography */
        --font-primary: 'Montserrat', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Apply the custom font to the entire page */
    body, .main, .stApp, [data-testid="stSidebar"] {
        font-family: var(--font-primary);
        background-color: var(--color-background);
        color: var(--color-text-dark);
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling with enhanced shadow and curves */
    .main-header {
        background: linear-gradient(135deg, var(--color-primary) 0%, #2a9d8f 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 128, 64, 0.2);
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin-top: 0;
        font-size: 1.1rem;
    }
    
    /* Sidebar styling with enhanced shadow */
    [data-testid=stSidebar] {
        background: linear-gradient(to bottom, #FFFFFF 0%, #FAFAFA 100%); /* Subtle gradient */
        border-right: 1px solid var(--color-border);
        padding: 1.5rem;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Ensure sidebar text is dark */
    [data-testid=stSidebar] * {
        color: var(--color-text-dark) !important;
        background-color: transparent !important;
    }
    
    [data-testid=stSidebar] .stSelectbox, 
    [data-testid=stSidebar] .stNumberInput {
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar title with separator */
    [data-testid=stSidebar] h3 {
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 0.7rem;
        margin-bottom: 1.5rem;
        color: var(--color-text-dark);
    }
    
    [data-testid=stSidebar] .stSelectbox label,
    [data-testid=stSidebar] .stNumberInput label {
        font-weight: 600;
        color: var(--color-text-dark);
        background-color: #f8f9fa; /* Gray background for labels */
        padding: 0.5rem 0.7rem;
        border-radius: 6px;
        display: block;
        margin-bottom: 0.4rem;
        border-left: 3px solid #adb5bd; /* Gray accent border */
    }
    
    /* Enhanced dropdown and input styling with gray borders */
    .stSelectbox, .stNumberInput, .stTextInput {
        color: var(--color-text-dark) !important;
        background-color: var(--color-background) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stSelectbox"] div[role="combobox"] {
        background-color: var(--color-background) !important;
        border-radius: 8px !important;
        border: 2px solid #ced4da !important; /* Gray border */
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSelectbox"] div[role="combobox"]:hover {
        border-color: #adb5bd !important; /* Lighter gray on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stSelectbox"] div[role="combobox"]:focus,
    [data-testid="stSelectbox"] div[role="combobox"]:active {
        border: 2px solid var(--color-primary) !important; /* Green border when focused */
        box-shadow: 0 0 0 3px rgba(0, 128, 64, 0.2) !important;
    }
    
    /* Input fields in sidebar */
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background-color: var(--color-background) !important;
        border: 2px solid #ced4da !important; /* Gray border */
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stNumberInput"] input:hover, [data-testid="stTextInput"] input:hover {
        border-color: #adb5bd !important; /* Lighter gray on hover */
    }
    
    [data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus {
        border: 2px solid var(--color-primary) !important; /* Green border when focused */
        box-shadow: 0 0 0 3px rgba(0, 128, 64, 0.2) !important;
    }
    
    /* Dropdown menu options */
    [data-baseweb="select"] > div {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    [data-baseweb="select"] span {
        color: var(--color-text-dark) !important;
    }
    
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    [data-testid="stSelectbox"] input {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Dropdown options in menu */
    [data-testid="stSelectbox"] [data-st-b] {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Container styling with enhanced shadow and smoother curves */
    .stContainer {
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: var(--color-background);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.3s ease;
    }
    
    .stContainer:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Metric card styling (following the reference style) */
    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        padding: 25px 30px;
        background-color: var(--color-background);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    .metric-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--color-text-dark);
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 3.5rem; /* Large and bold */
        font-weight: 700;
        line-height: 1.1;
        color: var(--color-text-dark);
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--color-primary); /* Use green for positive change */
        margin-top: 5px;
    }
    
    /* Alternative metric container styling */
    .metric-container {
        background: var(--color-background);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid var(--color-border);
        text-align: center;
    }
    
    .metric-value-alt {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--color-text-dark);
        margin: 0.2rem 0;
    }
    
    .metric-label-alt {
        font-size: 0.9rem;
        color: var(--color-text-light);
        margin: 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: var(--color-primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #006633; /* Darker green on hover */
    }
    
    /* Graph container styling with enhanced shadow and curves */
    .graph-container {
        background: var(--color-background);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--color-border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers with enhanced styling */
    h2 {
        color: var(--color-text-dark);
        border-bottom: 2px solid var(--color-primary);
        padding-bottom: 0.5rem;
        font-size: 1.8rem;
        border-radius: 8px;
    }
    
    h3, h4 {
        color: var(--color-text-dark);
    }
    
    /* Enhanced button styling with animation */
    .stButton>button {
        background: var(--color-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 128, 64, 0.2);
    }
    
    .stButton>button:hover {
        background: #006633; /* Darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 128, 64, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Data frame styling */
    .stDataFrame {
        border: 1px solid var(--color-border);
        border-radius: 12px;
    }
    
    /* Chart container with enhanced styling */
    .plot-container {
        background: var(--color-background);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid var(--color-border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar title */
    [data-testid=stSidebar] .stMarkdown h1,
    [data-testid=stSidebar] .stMarkdown h2,
    [data-testid=stSidebar] .stMarkdown h3 {
        color: var(--color-text-dark);
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--color-border);
        border-radius: 12px;
        overflow: hidden;
        background-color: var(--color-background);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Table styling */
    table {
        border-radius: 12px;
        background-color: var(--color-background);
    }
    
    th, td {
        color: var(--color-text-dark) !important;
        background-color: var(--color-background) !important;
    }
    
    /* Import Montserrat font properly */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
    
    /* Ensure Montserrat is loaded for all elements */
    * {
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Bar chart styling */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Main content background */
    section.main {
        background-color: #F8F8F8; /* Very slight off-white background */
    }
    
    /* Ensure all main content has white background and dark text */
    .main * {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Specific styling for Streamlit components */
    [data-testid="stColumn"], [data-testid="stContainer"], [data-testid="stExpander"] {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Chart container styling for Plotly */
    .stPlotlyChart, .stPlotlyChart > div, .stPlotlyChart > div > div {
        background-color: var(--color-background) !important;
    }
    
    /* Dropdown menu options */
    [data-baseweb="select"] > div {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    [data-baseweb="select"] span {
        color: var(--color-text-dark) !important;
    }
    
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    [data-testid="stSelectbox"] input {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Dropdown options in menu */
    [data-testid="stSelectbox"] [data-st-b] {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
    }
    
    /* Dropdown menu container */
    [data-testid="stSelectbox"] [data-baseweb="popover"] {
        border-radius: 8px !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Selected option in dropdown */
    [data-testid="stSelectbox"] [data-st-b] div[aria-selected="true"] {
        background-color: #f0f7f4 !important;
        color: var(--color-text-dark) !important;
        font-weight: 500;
    }
    
    /* Hover effect for options */
    [data-testid="stSelectbox"] [data-st-b] div:hover {
        background-color: #e6f2ee !important;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background-color: var(--color-background) !important;
        color: var(--color-text-dark) !important;
        border: 1px solid var(--color-border) !important;
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Dropdown options font */
    [data-testid="stSelectbox"] [data-st-b] {
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Button styling */
    button {
        background-color: var(--color-primary) !important;
        color: white !important;
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    button:hover {
        background-color: #006633 !important;
    }
    
    /* Ensure all Streamlit elements use Montserrat */
    .st-emotion-cache-16idsys p, .st-emotion-cache-16idsys div, .st-emotion-cache-16idsys span {
        font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# Add professional header
st.markdown('<div class="main-header"><h1>Trade Network Analytics Dashboard</h1><p>Visualize and analyze trade relationships and credit risk across your network</p></div>', unsafe_allow_html=True)


# Data loader
@st.cache_data()
def load_data(
    mode: Literal["invoices", "gnr", "random_k_out"] = "invoices",
):
    if mode == "invoices":
        return make_invoice_data()
    elif mode == "gnr":
        G = nx.gnr_graph(100, 0.8, seed=42)
        return G, None, None, None
    elif mode == "random_k_out":
        G = nx.generators.directed._random_k_out_graph_python(
            50, 5, alpha=0.5, self_loops=False, seed=42
        )
        return G, None, None, None

    else:
        raise NotImplementedError(f"Data generator `{mode}` supplied is not supported.")


G, df_relationships, df_nodes, df_invoices = load_data(mode="invoices")

# Initialize session state variables
if "previous_root_node" not in st.session_state:
    st.session_state.previous_root_node = None

# draw global graph
# fig, ax = plt.subplots()
# nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, ax=ax)
# st.pyplot(fig)


# Config sidebar
with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: var(--color-text-dark); margin-bottom: 1.5rem;'>Filters & Controls</h3>", unsafe_allow_html=True)
    st.selectbox(
        "Select Root Node:",
        list(G.nodes),
        key="root_node",
        placeholder=list(G.nodes)[0],
        help="Company Selection",
    )
    st.selectbox("Direction", ["out", "in", "both"], placeholder="out", key="direction")
    st.number_input("Degree of Separation:", 1, key="degree_of_sep")
    st.selectbox("Measure:", ["Credit Score", "Amount"], key="metric", disabled=True)


# Apply filtering to the graph based on configs

## 1. Start out with full graph
sub_G = G.copy()

## filter based on degree of separation and the selected node
if st.session_state.degree_of_sep and st.session_state.root_node is not None:
    root_node = st.session_state.root_node
    degree_of_sep = st.session_state.degree_of_sep
    successors = nx.dfs_successors(  # find successors with dept limit
        sub_G, root_node, depth_limit=degree_of_sep
    )
    nearby_nodes = [root_node]
    for successor_nodes in successors.values():
        # successor_nodes: list
        if successor_nodes:
            nearby_nodes.extend(successor_nodes)
    sub_G = sub_G.subgraph(nearby_nodes)

    # filter based on direction
    # if st.session_state.root_node is not None:
    #     root_node = st.session_state.root_node

    #     if st.session_state.direction == "out":
    #         selected_edges = [(u, v) for u, v in sub_G.edges if root_node == u]
    #         sub_G = sub_G.edge_subgraph(selected_edges)

    #     elif st.session_state.direction == "in":
    #         selected_edges = [(u, v) for u, v in sub_G.edges if root_node == v]
    #         sub_G = sub_G.edge_subgraph(selected_edges)

    #     elif st.session_state.direction == "both":
    #         selected_edges = [(u, v) for u, v in sub_G.edges if root_node in (u, v)]
    #         sub_G = sub_G.edge_subgraph(selected_edges)

    # pre-compute adjency matrix and y measure
    P = nx.attr_matrix(
        G,
        edge_attr="total_amount",
        # node_attr="credit_score",
        normalized=True,
        rc_order=list(G.nodes),
    )
    credit_scores = [score for node, score in G.nodes("credit_score")]
    company_idx = list(G.nodes).index(root_node)
    avg_credit_scores = P @ np.array(credit_scores)


# Network Chart
data = nx.cytoscape_data(sub_G)
left, right = st.columns(2)

# Enhanced stylesheet for more professional look using the new color palette
stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(id)",
            "width": 30,
            "height": 30,
            "shape": "ellipse",
            "background-color": "#008040",  # New primary color
            "color": "white",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "12px",
            "border-color": "#111111",  # Near black
            "border-width": 2,
            "border-opacity": 0.8,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "background-color": "#006633",  # Darker green
            "border-color": "#008040",
            "border-width": 3,
            "border-opacity": 1,
        },
    },
    {
        "selector": "edge",
        "style": {
            # "label": "data(weight)",
            "color": "#666",
            "line-color": "#999",
            "target-arrow-color": "#999",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "font-size": "10px",
            "text-rotation": "autorotate",
        },
    },
    {
        "selector": "edge:selected",
        "style": {
            "line-color": "#008040",
            "target-arrow-color": "#008040",
            "width": 3,
        },
    },
]

nodes = data["elements"]["nodes"]

# Default highlight selected node
for idx, node in enumerate(nodes):
    if node["data"]["id"] == root_node:
        node.update({"selected": True})
        nodes[idx] = node  # override
        break


# Make edges unselectable
_edges = data["elements"]["edges"]
edges = [{**e, "selectable": False} for e in _edges]

with left:
    st.markdown("## Trade Network")
    st.write(f"Root node: `{st.session_state.root_node}`")
    clicked_elements = cytoscape(
        dict(nodes=nodes, edges=edges),
        stylesheet,
        selection_type="single",
        user_panning_enabled=False,
        key="graph",
        layout={"name": "fcose", "animationDuration": 2},
        height="400px",
    )

    # if clicked_elements is not None:
    #     st.write(clicked_elements)

    with st.container(border=True):
        if st.session_state.graph.get("nodes"):
            # st.write(st.session_state.graph["nodes"][0])
            clicked_node = st.session_state.graph.get("nodes")[0]
            st.markdown(f"Selected node: `{clicked_node}`")
            clicked_node_attr = G.nodes(data=True)[clicked_node]
            st.write(clicked_node_attr)

            def update_root_node(new_node):
                """Update the root node to the clicked node"""
                # init previous_root_node if it does not exist yet
                if "previous_root_node" not in st.session_state:
                    st.session_state.previous_root_node = None
                # first, store current root node as previous, for backward navigation
                st.session_state.previous_root_node = st.session_state.root_node
                st.session_state.root_node = new_node

            st.button(
                "View Selected Node", on_click=update_root_node, args=(clicked_node,)
            )
            st.button(
                "View Previous Node",
                on_click=update_root_node,
                args=(st.session_state.previous_root_node,),
                disabled=False if st.session_state.get("previous_root_node") else True,
            )


with right:
    st.markdown(f"## {root_node} Insights")

    if st.session_state.graph.get("nodes"):
        clicked_node = st.session_state.graph.get("nodes")[0]

        # subset df_relationship depending on the selected direction
        match st.session_state.direction:
            case "in":
                q = "buyer_company == @root_node"
            case "out":
                q = "seller_company == @root_node"
            case "both":
                q = "buyer_company == @root_node or seller_company == @root_node"

        # subset dataframe and calculate metrics
        df_relationships_subset = df_relationships.query(q)
        df_relationships_subset = df_relationships_subset.assign(
            perc=lambda df: df["total_amount"] / df["total_amount"].sum()
        )
        company_metrics = calculate_company_metrics(df_relationships_subset)

        df_invoices_subset = df_invoices.query(q)

        # credit score breakdown
        with st.container(border=True):
            st.write("#### Credit Risk Breakdown")
            st.metric(
                "Average Buyer Credit Score",
                None
                if np.isnan(avg_credit_scores[company_idx])
                else round(avg_credit_scores[company_idx]),
            )

            _df = pd.DataFrame(
                list(sub_G.nodes(data="credit_score")),
                columns=["company", "credit_score"],
            )
            _df = (
                _df.loc[_df.company != root_node]
                .merge(
                    df_relationships_subset[
                        ["buyer_company", "total_amount", "perc"]
                    ].set_index("buyer_company"),
                    left_on="company",
                    right_index=True,
                )
                .sort_values("total_amount", ascending=False)
                .reset_index(drop=True)
            )

            # Enhanced matplotlib chart styling with new color palette
            if not _df.empty:  # Check if dataframe is not empty before plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = _df[["company", "perc"]].plot.barh(
                    y="perc",
                    x="company",
                    ax=ax,
                    xlabel="% Contribution by Transaction Value",
                    color='#008040',  # New primary color
                    edgecolor='white',
                    linewidth=1
                )
                ax.set_xlabel("% Contribution by Transaction Value", fontsize=12, fontweight='bold', color='#111111')
                ax.set_ylabel("Company", fontsize=12, fontweight='bold', color='#111111')
                ax.grid(axis='x', alpha=0.3)
                ax.set_axisbelow(True)
                ax.tick_params(colors='#111111')
                # Add value labels on bars
                for i, v in enumerate(_df['perc']):
                    ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=10)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No data available for the selected filters.")

            st.dataframe(
                _df,
                column_config={
                    "perc": st.column_config.NumberColumn("perc", format="%.2f%%")
                },
            )

        # transaction breakdown
        with st.container(border=True):
            st.write("#### Transaction History")
            # Use the new metric card grid layout
            st.markdown('<div class="metric-card-grid">', unsafe_allow_html=True)
            tl, tr = st.columns(2)
            bl, br = st.columns(2)
            
            # Enhanced metric cards with the new color palette
            with tl:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-header">{prettify_metric_name("nunique_buyers")}</p>
                    <p class="metric-value">{company_metrics[("nunique_buyers")]}</p>
                </div>
                """, unsafe_allow_html=True)
            with tr:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-header">{prettify_metric_name("nunique_sellers")}</p>
                    <p class="metric-value">{company_metrics[("nunique_sellers")]}</p>
                </div>
                """, unsafe_allow_html=True)
            with bl:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-header">{prettify_metric_name("total_invoices")}</p>
                    <p class="metric-value">{company_metrics[("total_invoices")]}</p>
                </div>
                """, unsafe_allow_html=True)
            with br:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-header">{prettify_metric_name("total_transaction_value")}</p>
                    <p class="metric-value">${company_metrics[("total_transaction_value")]:,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
            _df = (
                df_invoices_subset.groupby(df_invoices_subset.invoice_date.dt.date)[
                    ["invoice_id", "invoice_amount"]
                ]
                .agg(n=("invoice_id", "size"), total_amount=("invoice_amount", "sum"))
                .reset_index()
            )

            import plotly.express as px

            # Enhanced Plotly chart with new color palette and Montserrat font
            fig = px.bar(
                _df, 
                x="invoice_date", 
                y="total_amount", 
                title="Invoice History",
                color_discrete_sequence=['#008040']  # New primary color
            )
            fig.update_layout(
                title_font_size=16,
                title_font_color="var(--color-text-dark)",
                title_font_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                plot_bgcolor="var(--color-background)",
                paper_bgcolor="var(--color-background)",
                font_color="var(--color-text-dark)",
                font_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='var(--color-secondary-bar)',
                    title_font_color="var(--color-text-dark)",
                    title_font_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                    tickfont_color="var(--color-text-dark)",
                    tickfont_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='var(--color-secondary-bar)',
                    title_font_color="var(--color-text-dark)",
                    title_font_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                    tickfont_color="var(--color-text-dark)",
                    tickfont_family="Montserrat, 'Helvetica Neue', Arial, sans-serif",
                ),
                margin=dict(l=20, r=20, t=40, b=20)  # reduce margins for better fit
            )
            fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
            st.plotly_chart(fig, use_container_width=True)
            st.write(df_invoices_subset)
        # for metric in company_metrics:
        #     st.metric(prettify_metric_name(metric), company_metrics[metric])


# selected_node_in_graph = st.session_state.graph["nodes"][0]
# selected_node_in_graph
# fig, ax = plt.subplots()
# nx.draw_networkx(G, nx.spring_layout(G), ax=ax)
# st.pyplot(fig)