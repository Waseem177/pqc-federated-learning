#!/home/wasee/pqc-env/bin/python
"""Entry point: orchestrates the PQC-secured federated learning simulation."""

import sys
import warnings
warnings.filterwarnings("ignore")   # suppress liboqs version mismatch warning

_VENV_SITE = "/home/wasee/pqc-env/lib/python3.12/site-packages"
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

import argparse

from server import AggServer
from node import EdgeNode, RogueNode
from metrics import MetricsCollector
from dashboard import print_header, print_round_row, print_final_summary


def run_simulation(
    num_rounds: int = 10,
    num_nodes: int = 5,
    csv_out: str = "results.csv",
    attack: bool = False,
) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    server = AggServer()
    rogue_id = num_nodes - 1  # last node index is the attacker when --attack is set
    nodes = []
    for i in range(num_nodes):
        if attack and i == rogue_id:
            nodes.append(RogueNode(i, server.kem_public_key))
        else:
            nodes.append(EdgeNode(i, server.kem_public_key))
    for node in nodes:
        server.register_node(node.node_id, node.sig_public_key)

    collector = MetricsCollector()
    print_header(num_nodes, num_rounds, attack=attack)

    # ── Rounds ────────────────────────────────────────────────────────────────
    for rnd in range(num_rounds):
        updates = [node.prepare_update() for node in nodes]
        result  = server.aggregate(updates, round_num=rnd)
        summary = collector.record(result)
        print_round_row(summary)

    # ── Finish ────────────────────────────────────────────────────────────────
    collector.export_csv(csv_out)
    print_final_summary(collector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQC-Secured FL NIDS Simulator")
    parser.add_argument("--rounds", type=int, default=10,           help="Number of FL rounds (default 10)")
    parser.add_argument("--nodes",  type=int, default=5,            help="Number of edge nodes (default 5)")
    parser.add_argument("--csv",    type=str, default="results.csv", help="CSV output path")
    parser.add_argument("--attack", action="store_true",            help="Simulate rogue node: poisoned gradient + forged signature")
    args = parser.parse_args()

    run_simulation(num_rounds=args.rounds, num_nodes=args.nodes, csv_out=args.csv, attack=args.attack)
