"""
Zero-shot binary presence detection per CIFAR-10 class using CLIP.
For each class, the model answers: "is this class present in the image?"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from embeddings import load_embedder
from embeddings.prompts import build_class_text_embeddings

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 4
CACHE_DIR = Path("./embedding_cache")
THREASHOLD = 0.5051
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Image embedding cache ─────────────────────────────────────────────────────

def _cache_path(backend: str, model_name: str, pretrained: str) -> Path:
    key = f"{backend}__{model_name}__{pretrained}".replace("/", "-")
    return CACHE_DIR / f"{key}.pt"


@torch.no_grad()
def load_or_compute_image_embeddings(
    embedder,
    backend: str,
    model_name: str,
    pretrained: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (embeddings, labels) for the full CIFAR-10 test set, shape (N, d) and (N,).
    Loads from disk on repeat runs; encodes and saves on first run.
    """
    path = _cache_path(backend, model_name, pretrained)

    if path.exists():
        print(f"Loading cached image embeddings from {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data["embeddings"], data["labels"]

    print("Computing image embeddings (cached for future runs) ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=embedder.preprocess
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    all_embs, all_labels = [], []
    for images, labels in tqdm(loader, desc="Encoding images"):
        all_embs.append(embedder.encode_image(images).cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embs)   # (N, d)
    labels = torch.cat(all_labels)     # (N,)

    torch.save({"embeddings": embeddings, "labels": labels}, path)
    print(f"Saved to {path}")

    return embeddings, labels


# ── Per-class metrics ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_binary_classification(
    embedder,
    class_idx: int,
    img_embs: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Binary presence detection for one class using pre-computed image embeddings.
    """
    classname = CIFAR10_CLASSES[class_idx]

    pos_emb, neg_emb = build_class_text_embeddings(embedder, classname)
    text_embs = torch.stack([pos_emb.cpu(), neg_emb.cpu()], dim=0)  # (2, d)

    logits = img_embs @ text_embs.T                    # (N, 2)
    scores = logits.softmax(dim=-1)[:, 0]              # positive-class probability
    preds = scores > THREASHOLD
    binary_labels = labels == class_idx

    scores_np = scores.numpy()
    labels_np = binary_labels.numpy()

    tp = (preds & binary_labels).sum().item()
    fp = (preds & ~binary_labels).sum().item()
    fn = (~preds & binary_labels).sum().item()
    tn = (~preds & ~binary_labels).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    prec_c, rec_c, thresh_c = precision_recall_curve(labels_np, scores_np)
    # prec_c / rec_c have one extra sentinel point; thresholds is shorter by 1
    f1_c = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_c))

    return {
        "class":     classname,
        "accuracy":  (tp + tn) / len(labels),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc_score(labels_np, scores_np),
        "avg_prec":  average_precision_score(labels_np, scores_np),
        "best_threshold":  float(thresh_c[best_idx]),
        "best_f1_thresh":  float(f1_c[best_idx]),
        "best_prec_thresh": float(prec_c[best_idx]),
        "best_rec_thresh":  float(rec_c[best_idx]),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # kept for curve plotting
        "_scores": scores_np,
        "_labels": labels_np,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_curves(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.colormaps["tab10"].colors  # one colour per class

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_roc, ax_pr = axes

    fpr_grid = np.linspace(0, 1, 1000)
    rec_grid = np.linspace(0, 1, 1000)
    interp_tpr   = np.zeros_like(fpr_grid)
    interp_prec  = np.zeros_like(rec_grid)

    for r, color in zip(results, colors):
        scores, labels = r["_scores"], r["_labels"]
        name = r["class"]

        fpr, tpr, _ = roc_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=1.5, alpha=0.7,
                    label=f"{name}  (AUC={r['roc_auc']:.3f})")
        interp_tpr += np.interp(fpr_grid, fpr, tpr)

        prec, rec, _ = precision_recall_curve(labels, scores)
        ax_pr.plot(rec, prec, color=color, lw=1.5, alpha=0.7,
                   label=f"{name}  (AP={r['avg_prec']:.3f})")
        # precision_recall_curve returns decreasing recall — flip for interp
        interp_prec += np.interp(rec_grid, rec[::-1], prec[::-1])

    # macro-mean curves
    mean_tpr  = interp_tpr  / len(results)
    mean_prec = interp_prec / len(results)
    mean_auc  = np.mean([r["roc_auc"]  for r in results])
    mean_ap   = np.mean([r["avg_prec"] for r in results])

    ax_roc.plot(fpr_grid, mean_tpr, color="black", lw=2.5,
                label=f"mean  (AUC={mean_auc:.3f})")

    ax_pr.plot(rec_grid, mean_prec, color="black", lw=2.5,
               label=f"mean  (AP={mean_ap:.3f})")

    # best-F1 point on the mean PR curve
    mean_f1_grid = 2 * mean_prec * rec_grid / (mean_prec + rec_grid + 1e-8)
    bi = int(np.argmax(mean_f1_grid))
    br, bp, bf = rec_grid[bi], mean_prec[bi], mean_f1_grid[bi]
    ax_pr.plot(br, bp, "*", color="black", ms=14, zorder=5,
               label=f"best F1={bf:.3f}  (P={bp:.3f}, R={br:.3f})")
    ax_pr.annotate(
        f"F1={bf:.3f}\nP={bp:.3f}\nR={br:.3f}",
        xy=(br, bp), xytext=(br + 0.06, bp - 0.12),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="black"),
    )

    # ROC axes
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves — per class")
    ax_roc.legend(fontsize=8, loc="lower right")

    # PR axes
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curves — per class")
    ax_pr.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "roc_pr_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary CLIP presence detection on CIFAR-10"
    )
    parser.add_argument(
        "--backend", default="open_clip",
        choices=["open_clip", "openai_clip", "negation_clip"],
        help="CLIP library to use (default: open_clip)",
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model name — defaults: ViT-L-14 (open_clip), "
            "ViT-L/14 (openai_clip), ViT-B/32 (negation_clip)"
        ),
    )
    parser.add_argument(
        "--pretrained", default="openai",
        help="Pretrained weights tag for open_clip (default: openai)",
    )
    parser.add_argument(
        "--weights", default="negationclip_ViT-B32.pth",
        help="Path to fine-tuned .pth weights for negation_clip (default: negationclip_ViT-B32.pth)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete cached image embeddings and recompute",
    )
    parser.add_argument(
        "--plots-dir", default="./plots",
        help="Directory to save ROC / PR curve plots (default: ./plots)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _default_model = {"open_clip": "ViT-L-14", "openai_clip": "ViT-L/14", "negation_clip": "ViT-B/32"}
    model_name = args.model or _default_model[args.backend]
    # cache discriminator: pretrained tag for open_clip, weights stem for negation_clip, empty otherwise
    if args.backend == "open_clip":
        pretrained = args.pretrained
    elif args.backend == "negation_clip":
        pretrained = Path(args.weights).stem
    else:
        pretrained = ""

    if args.clear_cache:
        path = _cache_path(args.backend, model_name, pretrained)
        if path.exists():
            path.unlink()
            print(f"Cleared cache: {path}")

    print(f"Device   : {DEVICE}")
    print(f"Backend  : {args.backend}  model={model_name}")

    load_kwargs = {"device": DEVICE}
    if args.backend == "open_clip":
        load_kwargs["pretrained"] = args.pretrained
    elif args.backend == "negation_clip":
        load_kwargs["weights"] = args.weights

    embedder = load_embedder(backend=args.backend, model_name=model_name, **load_kwargs)

    img_embs, labels = load_or_compute_image_embeddings(
        embedder, args.backend, model_name, pretrained
    )

    results = [
        run_binary_classification(embedder, i, img_embs, labels)
        for i in tqdm(range(len(CIFAR10_CLASSES)), desc="Classifying")
    ]

    # ── Tabulate report ───────────────────────────────────────────────────────
    rows = [
        [
            r["class"],
            f"{r['accuracy']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['roc_auc']:.3f}",
            f"{r['avg_prec']:.3f}",
            r["tp"], r["fp"], r["fn"], r["tn"],
        ]
        for r in results
    ]

    def _mean(key):
        return np.mean([r[key] for r in results])

    rows.append([
        "mean (macro)",
        f"{_mean('accuracy'):.3f}",
        f"{_mean('precision'):.3f}",
        f"{_mean('recall'):.3f}",
        f"{_mean('f1'):.3f}",
        f"{_mean('roc_auc'):.3f}",
        f"{_mean('avg_prec'):.3f}",
        "", "", "", "",
    ])

    headers = ["Class", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "Avg-Prec", "TP", "FP", "FN", "TN"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ── Best-threshold report (PR curve, maximises F1) ────────────────────────
    thresh_rows = [
        [
            r["class"],
            f"{r['best_threshold']:.4f}",
            f"{r['best_f1_thresh']:.3f}",
            f"{r['best_prec_thresh']:.3f}",
            f"{r['best_rec_thresh']:.3f}",
        ]
        for r in results
    ]
    thresh_rows.append([
        "mean (macro)",
        f"{_mean('best_threshold'):.4f}",
        f"{_mean('best_f1_thresh'):.3f}",
        f"{_mean('best_prec_thresh'):.3f}",
        f"{_mean('best_rec_thresh'):.3f}",
    ])
    thresh_headers = ["Class", "Best Threshold", "F1", "Prec", "Rec"]
    print()
    print(tabulate(thresh_rows, headers=thresh_headers, tablefmt="rounded_outline"))

    plot_curves(results, Path(args.plots_dir))

    return results


if __name__ == "__main__":
    main()
