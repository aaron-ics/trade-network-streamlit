from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime

import networkx as nx
import numpy as np
import pandas as pd
from faker import Faker


# fix random seed
SEED = 42
Faker.seed(SEED)


@dataclass
class Invoice:
    invoice_id: str
    invoice_amount: int
    invoice_date: datetime.date
    buyer_company: str
    seller_company: str


class InvoiceSimulator:
    def __init__(self, n_companies: int = 10):
        # Init faker
        self.fake = Faker(locale="ID")

        # Init company universe
        self.companies = sorted(
            list(set(self.fake.company() for _ in range(n_companies)))
        )

        raw_weights = np.array(
            [self.fake.random_choices([1, 20], 1)[0] for company in self.companies]
        )

        _buyer_normalized_weights = (raw_weights / sum(raw_weights)).tolist()
        self.buyer_weights = OrderedDict(zip(self.companies, _buyer_normalized_weights))
        _seller_normalized_weights = ((1 / raw_weights) / sum(1 / raw_weights)).tolist()
        self.seller_weights = OrderedDict(
            zip(self.companies, _seller_normalized_weights)
        )

        # Assign credit scores
        score_range = (350, 850)
        self._credit_scores = [
            self.fake.random_int(*score_range, step=25)
            for _ in range(len(self.companies))
        ]

        # Credit score mapper
        self.company_credit_scores = dict(zip(self.companies, self._credit_scores))

        # invoice id
        self.invoice_counter = 1

    def _generate_invoice(self) -> Invoice:
        buyer_company = self.fake.random_elements(self.buyer_weights, 1)[0]
        seller_company = self.fake.random_elements(self.seller_weights, 1)[0]

        # if seller and buyer are the same company, resample from universe
        while True:
            if buyer_company != seller_company:
                break
            seller_company = self.fake.random_elements(
                self.seller_weights,
                1,
                use_weighting=True,
            )[0]

        invoice = Invoice(
            invoice_id="invoice_" + str(self.invoice_counter),
            invoice_amount=self.fake.random_int(100, 1000, step=50),
            invoice_date=self.fake.date_between(
                start_date=date(2025, 6, 1), end_date=date(2025, 6, 30)
            ),
            buyer_company=buyer_company,
            seller_company=seller_company,
        )
        # increment by 1
        self.invoice_counter += 1

        return invoice

    def generate_invoices(self, n: int = 5) -> list[Invoice]:
        return [self._generate_invoice() for _ in range(n)]


simulator = InvoiceSimulator(n_companies=10)


def load_invoices(**kwargs) -> list[Invoice]:
    """Simulate invoices"""
    global simulator

    invoices = simulator.generate_invoices(**kwargs)

    return invoices


def load_node_attrs() -> pd.DataFrame:
    """
    Make credit score in the format of:
    {
        "company 1": {"credit_score": 300},
        "company 2": {"credit_score": 350},
    }

    and the convert into pandas dataframe

    """

    nodes = {
        company: dict(credit_score=credit_score)
        for company, credit_score in simulator.company_credit_scores.items()
    }

    df_nodes = pd.DataFrame.from_dict(nodes, orient="index", dtype="int")
    return df_nodes


def aggregate_invoices_to_relationship(invoices: list[Invoice]) -> pd.DataFrame:
    """Aggregate invoices to relationship level"""
    df = pd.DataFrame(invoices)

    level = ["buyer_company", "seller_company"]
    df_relationships = (
        df.groupby(level)
        .agg(n=("invoice_id", "size"), total_amount=("invoice_amount", "sum"))
        .reset_index()
    )

    return df_relationships


def make_invoice_data(**kwargs):
    """Input a dataframe of buyer seller relationship
    This function will then translate that into node edge representation using networkX.
    Returns a tuple of data structures, containing:
        1. G: Graph representation of the companies' relational invoice data.
        2. df_relationships - Invoices aggregated to buyer-seller relationship level.
        3. df_nodes - Contains the node attributes of each company
        4. df_invoices - Contains the invoices data
    """
    invoices = load_invoices(n=50)
    df_invoices = pd.DataFrame(invoices).assign(
        invoice_date=lambda df: pd.to_datetime(df.invoice_date)
    )
    df_nodes: pd.DataFrame = load_node_attrs()
    df_relationships = aggregate_invoices_to_relationship(invoices)

    # Load edges into the graph
    G: nx.DiGraph = nx.from_pandas_edgelist(
        df_relationships,
        source="seller_company",
        target="buyer_company",
        edge_attr=True,
        create_using=nx.DiGraph(),  # Represent using directed graphs as invoices are directional in nature (i.e. seller sells to buyer)
    )

    # Add node attributes
    G.add_nodes_from((n, d.to_dict()) for n, d in df_nodes.iterrows())

    return G, df_relationships, df_nodes, df_invoices


def calculate_company_metrics(df_relationships):
    metrics = {
        "nunique_buyers": df_relationships["buyer_company"].nunique(),
        "nunique_sellers": df_relationships["seller_company"].nunique(),
        "total_invoices": df_relationships["n"].sum().item(),
        "total_transaction_value": df_relationships["total_amount"].sum().item(),
    }

    return metrics


if __name__ == "__main__":
    simulator = InvoiceSimulator(n_companies=10)
    invoices = simulator.generate_invoices(10)
    from pprint import pprint

    pprint(invoices)
