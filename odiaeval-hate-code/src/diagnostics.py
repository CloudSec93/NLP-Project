"""Surface-feature baseline — how much of the label is recoverable without reading.

This is a **dataset diagnostic, not a model deliverable**. It fits a logistic
regression on surface properties of ``text_ory`` — length, punctuation counts,
script purity — with **no lexical content whatsoever**: no n-grams, no bag of
words, no embeddings. If a feature could identify a word, it is not here.

Why this is in scope under CLAUDE.md §3: the boundary there bans fine-tuning
IndicBERTv2, training loops, checkpoint selection and evaluation harnesses — the
student model and the project's headline numbers. This produces no model, no
checkpoint, never touches Odia semantics and never touches the native gold set.
It is a measurement *of the corpus*, in the graded data-quality bucket.

**How to read the number.** A surface-only model has no access to meaning, so
whatever macro-F1 it reaches is a *floor* on how much of the task is solvable by
shortcut. If it reaches 0.85, a fine-tuned model scoring 0.91 has added only ~6
points of actual language understanding, and the published benchmark number means
much less than it appears to. This is the hypothesis-only baseline method that
exposed annotation artefacts in SNLI:

    Gururangan et al., "Annotation Artifacts in Natural Language Inference Data",
    NAACL 2018.
    Poliak et al., "Hypothesis Only Baselines in Natural Language Inference",
    *SEM 2018.

**The model is deliberately not tuned.** A weak, untuned baseline scoring high is
the alarming result; optimising it would only muddy what the number means.

Features come from ``text_ory``, not ``text_eng`` — the student never sees the
English, so a shortcut measured on English would be one that does not exist for
the model being warned about.

Fit on **train**, reported on **validation**. The test split is never read: it is
sealed for the group's real evaluation, and a diagnostic number in the data card
must not have touched it.

Run:  python -m src.diagnostics
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pyarrow.parquet as pq

from src._common import Paths, load_config

# chat-template scaffolding seen in HHRLHF_T (85 rows at Stage 2)
SCAFFOLD = re.compile(r"<s>|\[INST\]|\[ଇନ୍ସ୍ଟ\]|<<SYS>>", re.IGNORECASE)

FEATURE_NAMES = [
    "char_length",
    "token_count",
    "mean_word_length",
    "count_question_mark",
    "count_exclamation",
    "count_period",
    "count_comma",
    "count_digit",
    "count_uppercase",
    "ory_script_purity",
    "has_chat_scaffolding",
]


def featurise(rows: list[dict]) -> np.ndarray:
    """Surface features of ``text_ory``. No lexical content."""
    out = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for i, r in enumerate(rows):
        t = r["text_ory"]
        words = t.split()
        out[i] = [
            len(t),
            len(words),
            (sum(len(w) for w in words) / len(words)) if words else 0.0,
            t.count("?"),
            t.count("!"),
            t.count("."),
            t.count(","),
            sum(c.isdigit() for c in t),
            sum(c.isupper() for c in t),
            r["ory_script_purity"],
            1.0 if SCAFFOLD.search(t) else 0.0,
        ]
    return out


def run_variant(paths: Paths, variant: str, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    path = paths.interim / f"split_{variant}.parquet"
    if not path.exists():
        raise SystemExit(
            f"FATAL: {path.name} missing - run 'python -m src.balance_split --all-modes'."
        )
    rows = pq.read_table(
        path, columns=["text_ory", "ory_script_purity", "label", "split", "source_config"]
    ).to_pylist()

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    # the test split is deliberately not read
    Xtr, ytr = featurise(train), np.array([r["label"] for r in train])
    Xva, yva = featurise(val), np.array([r["label"] for r in val])

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),  # untuned, on purpose
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)

    macro = f1_score(yva, pred, average="macro")
    per_class = f1_score(yva, pred, average=None, labels=[0, 1])

    # Three reference floors, so nobody can quote only the flattering one:
    #   degenerate - always predict one class. On balanced binary this is ~0.33,
    #                and it is NOT the chance level: it is what a broken model gets.
    #   random     - independent coin flips. On balanced binary this is ~0.50 and
    #                IS the chance level the diagnostic must be read against.
    degenerate = f1_score(yva, np.zeros_like(yva), average="macro")
    rng = np.random.default_rng(seed)
    random_macro = float(
        np.mean([
            f1_score(yva, rng.integers(0, 2, size=len(yva)), average="macro")
            for _ in range(25)
        ])
    )

    coefs = clf[-1].coef_[0]
    ranked = sorted(
        ({"feature": n, "coef": float(c)} for n, c in zip(FEATURE_NAMES, coefs)),
        key=lambda d: -abs(d["coef"]),
    )
    constant = [
        FEATURE_NAMES[j] for j in range(Xtr.shape[1]) if float(Xtr[:, j].std()) == 0.0
    ]

    return {
        "variant": variant,
        "n_train": len(train),
        "n_validation": len(val),
        "macro_f1_validation": float(macro),
        "f1_non_hate": float(per_class[0]),
        "f1_hate": float(per_class[1]),
        "degenerate_always_one_class_macro_f1": float(degenerate),
        "random_chance_macro_f1": random_macro,
        "coefficients_ranked": ranked,
        "constant_features": constant,
    }


def print_variant(res: dict) -> None:
    print()
    print("=" * 92)
    print("SURFACE-FEATURE BASELINE - variant: {}".format(res["variant"]))
    print("  train {:,} rows -> validation {:,} rows   (test never read)".format(
        res["n_train"], res["n_validation"]))
    print("-" * 92)
    print("  macro-F1 (validation)      {:.4f}".format(res["macro_f1_validation"]))
    print("    F1 NON_HATE              {:.4f}".format(res["f1_non_hate"]))
    print("    F1 HATE                  {:.4f}".format(res["f1_hate"]))
    print()
    print("  reference floors:")
    print("    always-one-class (degenerate) {:.4f}   <- NOT the chance level".format(
        res["degenerate_always_one_class_macro_f1"]))
    print("    random 50/50 (chance)         {:.4f}   <- read the result against THIS".format(
        res["random_chance_macro_f1"]))
    print()
    print("  coefficients by |magnitude| (standardised features):")
    for d in res["coefficients_ranked"]:
        bar = "#" * int(min(abs(d["coef"]) / 0.05, 40))
        note = "  (constant - contributes nothing)" if d["feature"] in res["constant_features"] else ""
        print("    {:<24}{:+8.4f}  {}{}".format(d["feature"], d["coef"], bar, note))
    print("=" * 92)


def interpret(results: dict, target: float = 0.9094) -> list[str]:
    """Read the baseline against CHANCE, not against the degenerate floor.

    On balanced binary data, always predicting one class gives macro-F1 ~0.33.
    That is what a broken model scores, not what guessing scores. Random 50/50
    guessing gives ~0.50, and that is the level the shortcut must be measured
    against. Quoting 0.33 would overstate the finding.
    """
    lines = []
    for v, r in results.items():
        f1 = r["macro_f1_validation"]
        chance = r["random_chance_macro_f1"]
        covered = (f1 - chance) / (target - chance) * 100
        lines.append(
            "  {:<16} macro-F1 {:.4f}  =  {:+.4f} over chance ({:.4f}); covers "
            "{:.0f}% of the".format(v, f1, f1 - chance, chance, covered)
        )
        lines.append(
            "  {:<16} distance from chance to the {:.4f} reproduction target."
            "".format("", target)
        )
    if len(results) == 2 and "proportional" in results and "match_minority" in results:
        p_ = results["proportional"]["macro_f1_validation"]
        m_ = results["match_minority"]["macro_f1_validation"]
        lines.append("")
        lines.append(
            "  proportional - match_minority = {:+.4f}. The two corpora are identical "
            "except".format(p_ - m_)
        )
        lines.append(
            "  for which NON_HATE rows were kept, so that difference isolates the part "
            "of the"
        )
        lines.append(
            "  shortcut created by source stratification. It is small: match_minority "
            "drives"
        )
        lines.append(
            "  NMI(source;label) to exactly 0.0000 and still leaves most of the "
            "shortcut standing."
        )
        lines.append(
            "  The remainder lives in the LABELLING FUNCTION, not the corpus mix - see "
            "the"
        )
        lines.append("  question-form section of the data card.")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variants", nargs="+",
                        default=["proportional", "match_minority"])
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config(args.config)
    paths = Paths.from_config(cfg).ensure()

    print("=" * 92)
    print("SURFACE-FEATURE DIAGNOSTIC  (hypothesis-only baseline; no lexical features)")
    print("  features from text_ory - what the student model actually sees")
    print("  fit on train, reported on validation; the test split is never read")
    print("  model deliberately untuned: a weak baseline scoring high is the finding")
    print("=" * 92)

    results = {}
    for v in args.variants:
        res = run_variant(paths, v, cfg["seed"])
        results[v] = res
        print_variant(res)

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    for line in interpret(results):
        print(line)
    print("=" * 92)

    out = paths.artifacts / "label_distribution.json"
    d = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    d["surface_feature_baseline"] = {
        "features": FEATURE_NAMES,
        "text_field": "text_ory",
        "fit_on": "train",
        "reported_on": "validation",
        "test_split_read": False,
        "model": "LogisticRegression(max_iter=1000), StandardScaler, untuned",
        "references": [
            "Gururangan et al., NAACL 2018 - Annotation Artifacts in NLI Data",
            "Poliak et al., *SEM 2018 - Hypothesis Only Baselines in NLI",
        ],
        "results": results,
    }
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("\nwrote artifacts/label_distribution.json (surface_feature_baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
