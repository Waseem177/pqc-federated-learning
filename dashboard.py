#!/usr/bin/env python3
"""Terminal dashboard: rich-based live per-round table + final PQC-vs-RSA summary."""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

_table: Table | None = None
_num_nodes: int = 5
_num_rounds: int = 10
def print_header(num_nodes: int, num_rounds: int, attack: bool = False) -> None:
    global _table, _num_nodes, _num_rounds
    _num_nodes = num_nodes
    _num_rounds = num_rounds

    console.print()

    title_text = Text()
    title_text.append("PQC-Secured Federated Learning NIDS", style="bold white")
    title_text.append("  ·  Live Simulation", style="dim white")

    config_text = Text(justify="center")
    config_text.append(f"{num_nodes} edge nodes", style="bold cyan")
    config_text.append("  ·  ", style="dim")
    config_text.append(f"{num_rounds} FL rounds", style="bold cyan")
    config_text.append("  ·  ", style="dim")
    config_text.append("Kyber512", style="bold green")
    config_text.append(" + ", style="dim")
    config_text.append("ML-DSA-44", style="bold green")
    config_text.append(" + ", style="dim")
    config_text.append("AES-GCM", style="bold green")

    console.print(Panel(
        config_text,
        title=title_text,
        subtitle="[dim]Post-Quantum Cryptography Benchmark[/dim]",
        border_style="bright_blue",
        padding=(0, 2),
        expand=True,
    ))

    if attack:
        attack_text = Text(justify="center")
        attack_text.append("ATTACK MODE ENABLED", style="bold red")
        attack_text.append("  ·  ", style="dim red")
        attack_text.append(f"Node {num_nodes - 1}", style="bold red")
        attack_text.append(" is a rogue node — poisoned gradients + forged ML-DSA-44 signature", style="red")
        console.print(Panel(
            attack_text,
            title="[bold red]Rogue Node Simulation[/bold red]",
            border_style="red",
            padding=(0, 2),
            expand=True,
        ))

    console.print()

    _table = Table(
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold white",
        show_lines=False,
        expand=False,
        padding=(0, 1),
        title="[bold white]Round-by-Round PQC Metrics[/bold white]",
        title_style="bold",
        caption="[dim]Round 1 = warmup (JIT + AES context init)[/dim]",
    )

    _table.add_column("Round",      style="white",        justify="right",  min_width=6)
    _table.add_column("Nodes",      style="cyan",         justify="right",  min_width=6)
    _table.add_column("Avg PQC ms", style="green",        justify="right",  min_width=11)
    _table.add_column("Enc ms",     style="yellow",       justify="right",  min_width=9)
    _table.add_column("Dec ms",     style="yellow",       justify="right",  min_width=9)
    _table.add_column("OH KB/node", style="blue",         justify="right",  min_width=11)
def print_round_row(summary) -> None:
    oh_kb = summary.total_overhead_bytes / summary.num_nodes / 1024 if summary.num_nodes else 0.0
    is_warmup = summary.round_num == 0

    round_label = Text(str(summary.round_num + 1), justify="right")
    if is_warmup:
        round_label.stylize("bold yellow")
        round_label.append("  ⚑", style="dim yellow")

    row_style = "on grey7" if is_warmup else ""

    _table.add_row(
        round_label if is_warmup else str(summary.round_num + 1),
        str(summary.num_nodes),
        f"{summary.avg_pqc_ms:.3f}",
        f"{summary.total_enc_ms:.2f}",
        f"{summary.total_dec_ms:.2f}",
        f"{oh_kb:.1f}",
        style=row_style,
    )

    for nid in summary.rejected_nodes:
        _table.add_row(
            Text("  ↳", style="bold red"),
            Text(f"Node {nid} REJECTED", style="bold red"),
            Text("—", style="dim red"),
            Text("—", style="dim red"),
            Text("—", style="dim red"),
            Text("FORGED SIG", style="bold red"),
            style="on dark_red",
        )

    sys.stdout.flush()
def print_final_summary(collector) -> None:
    from metrics import RSA_BASELINE_MS, PQC_LABEL

    # Print the accumulated table now
    console.print(_table)
    console.print()

    avg_pqc   = collector.avg_pqc_ms()
    speedup   = collector.speedup_vs_rsa()
    oh_total  = collector.total_overhead_bytes()
    oh_per_nd = collector.per_node_overhead_kb()
    rounds    = len(collector)
    nodes     = collector._rounds[0].num_nodes if collector._rounds else 0

    # Left panel: PQC results
    pqc_lines = Text()
    pqc_lines.append("Algorithm Stack\n", style="bold underline cyan")
    pqc_lines.append("  KEM:    ", style="dim"); pqc_lines.append("Kyber512 (ML-KEM)\n", style="green")
    pqc_lines.append("  Auth:   ", style="dim"); pqc_lines.append("ML-DSA-44 (Dilithium)\n", style="green")
    pqc_lines.append("  Cipher: ", style="dim"); pqc_lines.append("AES-256-GCM\n\n", style="green")
    pqc_lines.append("Performance\n", style="bold underline cyan")
    pqc_lines.append(f"  Rounds:       ", style="dim"); pqc_lines.append(f"{rounds}\n", style="white")
    pqc_lines.append(f"  Nodes/round:  ", style="dim"); pqc_lines.append(f"{nodes}\n", style="white")
    pqc_lines.append(f"  Avg PQC/node: ", style="dim"); pqc_lines.append(f"{avg_pqc:.4f} ms\n", style="bold green")
    pqc_lines.append(f"  PQC overhead: ", style="dim"); pqc_lines.append(f"{oh_per_nd:.1f} KB/update\n", style="blue")
    pqc_lines.append(f"  Total OH:     ", style="dim"); pqc_lines.append(f"{oh_total / 1024:.1f} KB\n", style="blue")

    # Right panel: comparison
    cmp_lines = Text()
    cmp_lines.append("Baseline (RSA-2048)\n", style="bold underline yellow")
    cmp_lines.append(f"  Avg round:    ", style="dim"); cmp_lines.append(f"{RSA_BASELINE_MS:.2f} ms\n", style="yellow")
    cmp_lines.append(f"  Algorithm:    ", style="dim"); cmp_lines.append("RSA-2048 + SHA-256\n\n", style="yellow")
    cmp_lines.append("PQC Result\n", style="bold underline green")
    cmp_lines.append(f"  Avg round:    ", style="dim"); cmp_lines.append(f"{avg_pqc:.4f} ms\n", style="bold green")
    cmp_lines.append(f"  Speedup:      ", style="dim")
    cmp_lines.append(f"{speedup:.1f}× faster", style="bold bright_green")
    cmp_lines.append("\n\n")
    cmp_lines.append("Output\n", style="bold underline cyan")
    cmp_lines.append(f"  CSV:          ", style="dim"); cmp_lines.append("results.csv\n", style="cyan")

    pqc_panel = Panel(
        pqc_lines,
        title="[bold cyan]PQC Metrics[/bold cyan]",
        border_style="cyan",
        padding=(0, 2),
    )
    rsa_panel = Panel(
        cmp_lines,
        title="[bold yellow]RSA Comparison[/bold yellow]",
        border_style="yellow",
        padding=(0, 2),
    )

    console.print(Columns([pqc_panel, rsa_panel], equal=True, expand=True))
    console.print()

    verdict = Text(justify="center")
    verdict.append("PQC is NIST-standardized quantum-resistant  ·  ", style="dim")
    verdict.append(f"{speedup:.0f}× faster than RSA", style="bold bright_green")
    verdict.append("  ·  ready for edge deployment", style="dim")
    console.print(Panel(verdict, border_style="bright_green", padding=(0, 2)))
    console.print()
    sys.stdout.flush()
