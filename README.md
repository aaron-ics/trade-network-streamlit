# Trade Network Streamlit Playground

Exploratory Streamlit application that visualises fictitious buyer-seller relationships using invoice simulations. The goal is to experiment with network analysis techniques for credit scoring by combining **NetworkX** for graph modelling and **Streamlit** (with `streamlit-d3graph` and `st-cytoscape`) for interactive exploration.

## Features
- Simulate invoice activity between synthetic companies with controllable randomness.
- Build a directed trade network with NetworkX, enriched with credit score attributes.
- Interactively inspect ego networks, relationship volumes, and credit risk indicators within Streamlit.
- Overlay quick analytics such as buyer concentration, invoice history, and average partner credit scores.
- Swap between different graph generators (simulated invoices, random k-out, or GNR preferential attachment) for experimentation.

## Project Layout
- `app.py` – Streamlit interface for configuring and exploring the trade network.
- `src/invoices/invoice_simulator.py` – Invoice generator, relationship aggregation, and helper metrics.
- `utils.py` – Presentation helpers used in the UI.
- `nb/` – Notebooks for NetworkX experiments and analysis (see `nb/README.md`).

## Getting Started
1. **Install uv (if you don't already have it)**
   From the official [uv install guide](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Create the environment and install dependencies**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

## Running the App
```bash
uv run streamlit run app.py
```
The app launches at `http://localhost:8501` by default. Use the sidebar controls to choose the root company, direction of analysis, and degree of separation. Clicking a node in the graph drills into its relationships and supporting invoice metrics.

## Customising the Simulation
- Adjust the number of companies, invoice volume, or credit score ranges inside `src/invoices/invoice_simulator.py`.
- Extend `load_data` in `app.py` to plug in real data sources or additional graph generators.
- Modify the Streamlit containers to surface new metrics or visualisations (e.g. centrality scores, anomaly alerts).

## Development Notes
- The project targets **Python 3.12+**.
- Dependencies are managed via `pyproject.toml`; the repo includes a `uv.lock` if you prefer [uv](https://github.com/astral-sh/uv) for reproducible environments.
- The Streamlit session state stores the currently selected node to keep the graph and metrics panes in sync.
- A future direction is to integrate real invoice datasets and experiment with automated credit scoring heuristics on top of the network structure.

## Acknowledgements
Built with Streamlit, NetworkX, `streamlit-d3graph`, `st-cytoscape`, Plotly, and Faker for synthetic data.
