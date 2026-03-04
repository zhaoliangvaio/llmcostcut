"""
AG-News + GCP example for LLMCostCut.

End-to-end example: LLMCostCut with GCP classifier on the AG-News dataset,
with full concept-label supervision at every DAG node.

Concept labels
--------------
GCPClassifier uses a DAG of intermediate concept nodes before the final
prediction. Each node has its own linear head (concept_predictors) and can
be trained with a dedicated label. For AG-News we use 24 binary concept
nodes plus one sink (25 nodes total). Each concept answers: "Does the text
mention or discuss X?" (true/false).

  Nodes 0–3   – entities (person, org, location, country)
  Nodes 4–7   – business (company, stock/market, trade/deal, economy)
  Nodes 8–11  – sport (sport, team/league, match/score, player)
  Nodes 12–15 – tech (technology, product/device, research/science, internet/software)
  Nodes 16–19 – governance/events (government/policy, election/vote, conflict/war, disaster/crime)
  Nodes 20–23 – other (health/medicine, entertainment, energy/environment, space/weather)
  Node 24 (sink) – final 4-class topic (World / Sports / Business / Sci/Tech)

Training uses forward_with_concepts() and a weighted cross-entropy loss at
every node: L = L_final + w * (L_node0 + ... + L_node23).

What this script does
---------------------
  1. load_agnews()       – load AG-News (train only, optional n_samples)
  2. inspect_concept_nodes() – forward_with_concepts() and per-node confidence
  3. run_online()       – online distillation
  4. run_offline()      – offline budget labelling

Run:
  python examples/example.py
  python examples/example.py --n_samples 2000
"""

import argparse
import csv
import os
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch

from llmcostcut.monitor import monitor, _registry
from llmcostcut.defaults import get_device, get_encoder, get_tokenizer

torch.autograd.set_detect_anomaly(True)

# ---------------------------------------------------------------------------
# AG-News constants
# ---------------------------------------------------------------------------

TOPIC_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
TOPIC_TASK = {"topic": TOPIC_CLASSES}

# 24 binary concepts + 1 sink = 25 GCP nodes
AGNEWS_CONCEPTS = [
    "mentions_person", "mentions_org", "mentions_location", "mentions_country",
    "mentions_company", "mentions_stock_or_market", "mentions_trade_or_deal", "mentions_economy",
    "mentions_sport", "mentions_team_or_league", "mentions_match_score", "mentions_player",
    "mentions_technology", "mentions_product_or_device", "mentions_research_or_science", "mentions_internet_or_software",
    "mentions_government_or_policy", "mentions_election_or_vote", "mentions_conflict_or_war", "mentions_disaster_or_crime",
    "mentions_health_or_medicine", "mentions_entertainment", "mentions_energy_or_environment", "mentions_space_or_weather",
]
NUM_CONCEPT_NODES = len(AGNEWS_CONCEPTS)
SINK_NODE_IDX = NUM_CONCEPT_NODES
NUM_GCP_NODES = SINK_NODE_IDX + 1

CONCEPT_DESCRIPTIONS = {i: name for i, name in enumerate(AGNEWS_CONCEPTS)}
CONCEPT_DESCRIPTIONS[SINK_NODE_IDX] = "topic"

# ---------------------------------------------------------------------------
# GCP DAG configuration
# ---------------------------------------------------------------------------

GCP_AGNEWS_24CONCEPTS = {
    "edges": [(i, SINK_NODE_IDX) for i in range(NUM_CONCEPT_NODES)],
    "concept_dim": 256,
    "use_resnet": True,
    "dropout": 0.1,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("AGNEWS_CACHE_DIR", Path.home() / ".cache" / "agnews"))
_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}
_CSV_INT2STR = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_agnews(split: str = "train", max_samples: int = None, seed: int = 42) -> list:
    """Return [(text, topic_label), ...] from AG-News. Uses HuggingFace datasets; raises on load failure."""
    data = _try_hf(split) or _download_csv(split)
    if max_samples and max_samples < len(data):
        random.Random(seed).shuffle(data)
        data = data[:max_samples]
    return data


def _try_hf(split: str) -> list:
    try:
        from datasets import load_dataset
        hf_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        ds = load_dataset("ag_news", split=split)
        return [(r["text"], hf_map[r["label"]]) for r in ds]
    except Exception as e:
        raise RuntimeError(f"Failed to load AG-News via HuggingFace datasets: {e}") from e


def _download_csv(split: str) -> list:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}.csv"
    if not cache_file.exists():
        url = _CSV_URLS[split]
        print(f"[agnews] Downloading {split} split …")
        urllib.request.urlretrieve(url, cache_file)
        print(f"[agnews] Saved → {cache_file}")
    data = []
    with cache_file.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            label_str = _CSV_INT2STR.get(int(row[0]))
            text = (row[1].strip() + " " + row[2].strip()).strip()
            if label_str and text:
                data.append((text, label_str))
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset():
    _registry.tasks.clear()


def print_header(title: str):
    bar = "─" * 64
    print(f"\n┌{bar}┐")
    print(f"│  {title:<62}│")
    print(f"└{bar}┘")


def print_stats(label: str, total: int, fallbacks: int, elapsed_s: float):
    rate = fallbacks / total * 100 if total else 0.0
    msg = f"  {label:<34} samples={total:>5}  fallbacks={fallbacks:>5}  LLM-rate={rate:5.1f}%  time={elapsed_s:.1f}s"
    print(msg)


# ---------------------------------------------------------------------------
# Demo 1: Online distillation with concept supervision
# ---------------------------------------------------------------------------

def run_online(data: list, p_threshold: float = 0.8, n_samples: int = None):
    """Stream samples through monitor() with concept supervision."""
    print_header("Demo 1 – Online Distillation with Concept Supervision")
    reset()
    if n_samples is None:
        n_samples = len(data)
    data_sub = data[:n_samples]
    # concept_info: task_id -> list of per-node description strings (LLM gets one 0/1 per node).
    # Use the 24 concept names so the teacher returns topic__node_0..topic__node_23.
    llm_kwargs = {
        "concept_info": {
            "topic": AGNEWS_CONCEPTS,
        }
    }
    classifier_kwargs = {
        "edges": [(i, SINK_NODE_IDX) for i in range(NUM_CONCEPT_NODES)],
        "concept_dim": 256,
        "use_resnet": True,
        "dropout": 0.1
    }
    print(f"\n  [data] {len(data_sub)} samples  p_threshold={p_threshold}")

    window = max(1, len(data_sub) // 10)
    win_counts = defaultdict(int)
    win_fall = defaultdict(int)
    win_correct = defaultdict(int)
    fallbacks = 0
    correct = 0
    t0 = time.perf_counter()

    for i, (text, label) in enumerate(data_sub):

        # uncomment to run GCP mode
        # pred_result, fb = monitor(
        #     TOPIC_TASK, text, llm_fn=None,
        #     mode="online",
        #     device=1,
        #     classifier_type="gcp",
        #     classifier_kwargs=classifier_kwargs,
        #     llm_kwargs=llm_kwargs,
        # )
        
        pred_result, fb = monitor(
            TOPIC_TASK, text,
            mode="online",
            device="cuda:1" if torch.cuda.is_available() else "cpu",
            classifier_type="deep_mlp",
        )
        pred = pred_result.get("topic")
        if pred == label:
            correct += 1
        if fb:
            fallbacks += 1
        w = i // window
        win_counts[w] += 1
        win_fall[w] += int(fb)
        if pred == label:
            win_correct[w] += 1

    accuracy = correct / len(data_sub) * 100 if data_sub else 0.0
    print(f"\n  Accuracy: {correct}/{len(data_sub)} = {accuracy:.1f}%")
    print("\n  Fallback rate per decile:")
    for w in sorted(win_counts):
        rate = win_fall[w] / win_counts[w] * 100
        bar = "█" * int(rate / 5)
        print(f"    decile {w+1:>2}  {rate:5.1f}%  {bar}")
    print("\n  Accuracy per decile:")
    for w in sorted(win_counts):
        acc_w = win_correct[w] / win_counts[w] * 100 if win_counts[w] else 0.0
        bar = "█" * int(acc_w / 5)
        print(f"    decile {w+1:>2}  {acc_w:5.1f}%  {bar}")
    print_stats("done", len(data_sub), fallbacks, time.perf_counter() - t0)
    monitor.close()


def run_offline(data: list, p_threshold: float = 0.8):
    """Stream samples through monitor() with concept supervision."""
    print_header("Demo 1 – Online Distillation with Concept Supervision")
    reset()
    # concept_info: task_id -> list of per-node description strings (LLM gets one 0/1 per node).
    # Use the 24 concept names so the teacher returns topic__node_0..topic__node_23.
    llm_kwargs = {
        "concept_info": {
            "topic": AGNEWS_CONCEPTS,
        }
    }
    classifier_kwargs = {
        "edges": [(i, SINK_NODE_IDX) for i in range(NUM_CONCEPT_NODES)],
        "concept_dim": 256,
        "use_resnet": True,
        "dropout": 0.1
    }
    print(f"\n  [data] {len(data)} samples  p_threshold={p_threshold}")

    window = max(1, len(data) // 10)
    win_counts = defaultdict(int)
    win_fall = defaultdict(int)
    win_correct = defaultdict(int)
    fallbacks = 0
    correct = 0
    t0 = time.perf_counter()

    fb_cnt = 0
    total_cnt = 0

    for i in range(100):
        pred_result, fb = monitor(
            TOPIC_TASK, data, llm_fn=None,
            mode="offline",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            classifier_type="gcp",
            classifier_kwargs=classifier_kwargs,
            llm_kwargs=llm_kwargs,
            offline_select_method="random",
            offline_select_budget=10
        )
        total_cnt += len(fb)
        fb_cnt += sum(fb)
    
    print(f"Fallback rate: {fb_cnt / total_cnt * 100:.1f}%")

    monitor.close()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLMCostCut + GCP with concept labels on AG-News")
    parser.add_argument("--n_samples", type=int, default=None, help="Max samples (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    print("Loading AG-News (train only) …")
    t0 = time.perf_counter()
    data = load_agnews("train", max_samples=args.n_samples, seed=args.seed)
    print(f"  samples={len(data):,}  ({time.perf_counter()-t0:.1f}s)\n")
    print("Concept nodes: " + ", ".join(CONCEPT_DESCRIPTIONS[i] for i in range(min(5, NUM_GCP_NODES))) + " ...")

    run_online(
        data,
        p_threshold=args.threshold,
        n_samples=args.n_samples
    )

    # uncomment to run offline mode
    # run_offline(
    #     data,
    #     p_threshold=args.threshold,
    # )


if __name__ == "__main__":
    main()
