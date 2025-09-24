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
    st.set_page_config(layout="wide")


wide_space_default()


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

# draw global graph
# fig, ax = plt.subplots()
# nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, ax=ax)
# st.pyplot(fig)


# Config sidebar
with st.sidebar:
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
stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(id)",
            "width": 20,
            "height": 20,
            "shape": "circle",
            "color": "white",
        },
    },
    {
        "selector": "edge",
        "style": {
            # "label": "sold_to",
            "color": "white",
            # "labelwidth": 0.15,
            "labelwidth": "mapData()",
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-style": "dotted",
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

            fig, ax = plt.subplots()
            _df[["company", "perc"]].plot.barh(
                y="perc",
                x="company",
                ax=ax,
                xlabel="% Contribution by Transaction Value",
            )
            st.pyplot(fig)

            st.dataframe(
                _df,
                column_config={
                    "perc": st.column_config.NumberColumn("perc", format="percent")
                },
            )

        # transaction breakdown
        with st.container(border=True):
            st.write("#### Transaction History")
            tl, tr = st.columns(2)
            bl, br = st.columns(2)
            tl.metric(
                prettify_metric_name("nunique_buyers"),
                company_metrics[("nunique_buyers")],
            )
            tr.metric(
                prettify_metric_name("nunique_sellers"),
                company_metrics[("nunique_sellers")],
            )
            bl.metric(
                prettify_metric_name("total_invoices"),
                company_metrics[("total_invoices")],
            )
            br.metric(
                prettify_metric_name("total_transaction_value"),
                company_metrics[("total_transaction_value")],
            )
            _df = (
                df_invoices_subset.groupby(df_invoices_subset.invoice_date.dt.date)[
                    ["invoice_id", "invoice_amount"]
                ]
                .agg(n=("invoice_id", "size"), total_amount=("invoice_amount", "sum"))
                .reset_index()
            )

            import plotly.express as px

            fig = px.bar(
                _df, x="invoice_date", y="total_amount", title="Invoice History"
            )
            st.plotly_chart(fig)
            st.write(df_invoices_subset)
        # for metric in company_metrics:
        #     st.metric(prettify_metric_name(metric), company_metrics[metric])


# selected_node_in_graph = st.session_state.graph["nodes"][0]
# selected_node_in_graph
# fig, ax = plt.subplots()
# nx.draw_networkx(G, nx.spring_layout(G), ax=ax)
# st.pyplot(fig)
