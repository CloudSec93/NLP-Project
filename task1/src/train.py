"""Fine-tune IndicBERTv2-MLM-only on one variant of the Task 1 benchmark release.

Assignment constraints enforced here:

* training uses the train split only;
* the checkpoint is selected on validation macro-F1;
* the benchmark test split and the native Odia gold set are never loaded.

Run one variant:

    python -m src.train --variant proportional

Both variants are two independent runs, which is what the data card asks for.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np

from .common import (
    REPO_ROOT,
    compute_metrics,
    environment_fingerprint,
    load_config,
    pick_precision,
    resolve_model_revision,
    resolve_path,
    set_seed,
    write_json,
)
from .data import assert_no_split_leakage, load_benchmark, to_model_frame, tokenize


def parse_args(argv=None):
    cfg = load_config()
    t = cfg["training"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--variant", required=True, choices=["proportional", "match_minority"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", default=None, help="defaults to models/<variant>")
    p.add_argument("--model-name", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=t["seed"])
    p.add_argument("--precision", default=None, choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-train-samples", type=int, default=None,
        help="smoke-test switch: truncate the train split to N rows",
    )
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def _training_arguments(**kwargs):
    """Build TrainingArguments across the 4.4x rename of evaluation_strategy."""
    from transformers import TrainingArguments

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in accepted and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in accepted and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unknown = [k for k in kwargs if k not in accepted]
    for k in unknown:
        kwargs.pop(k)
    return TrainingArguments(**kwargs), unknown


def main(argv=None) -> Path:
    args = parse_args(argv)
    cfg = load_config(args.config)
    tcfg, mcfg, tokcfg = cfg["training"], cfg["model"], cfg["tokenizer"]

    model_name = args.model_name or mcfg["name"]
    data_root = args.data_root or cfg["data"]["root"]
    max_length = args.max_length or tokcfg["max_length"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    lr = args.lr or tcfg["learning_rate"]
    train_bs = args.batch_size or tcfg["train_batch_size"]
    eval_bs = args.eval_batch_size or tcfg["eval_batch_size"]
    grad_accum = args.grad_accum or tcfg["gradient_accumulation_steps"]
    precision = args.precision or tcfg["mixed_precision"]

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / cfg["paths"]["models_dir"] / args.variant
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
    )

    # ---------------------------------------------------------------- data
    frames = {s: load_benchmark(data_root, args.variant, s) for s in ("train", "validation")}
    leakage = assert_no_split_leakage(frames)

    train_df = to_model_frame(frames["train"])
    val_df = to_model_frame(frames["validation"])
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if args.max_eval_samples:
        val_df = val_df.sample(n=min(args.max_eval_samples, len(val_df)), random_state=args.seed)

    revision = mcfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    train_ds = tokenize(train_df, tokenizer, max_length, padding=tokcfg["padding"])
    val_ds = tokenize(val_df, tokenizer, max_length, padding=tokcfg["padding"])

    # --------------------------------------------------------------- model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=mcfg["num_labels"],
        id2label={0: "NON_HATE", 1: "HATE"},
        label2id={"NON_HATE": 0, "HATE": 1},
    )

    def _metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        m = compute_metrics(labels, preds)
        # Trainer only needs scalars for checkpoint selection.
        return {k: v for k, v in m.items() if isinstance(v, (int, float))}

    targs, dropped = _training_arguments(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        max_grad_norm=tcfg["max_grad_norm"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        evaluation_strategy=tcfg["evaluation_strategy"],
        save_strategy=tcfg["save_strategy"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        save_total_limit=tcfg["save_total_limit"],
        logging_steps=tcfg["logging_steps"],
        dataloader_num_workers=args.num_workers,
        save_safetensors=False,
        report_to=[],
        **pick_precision(precision),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics,
    )

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    # The best checkpoint is already loaded; persist it as the run's model.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    final_val = trainer.evaluate()
    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h]

    run_config = {
        "task": "hate-speech detection (OdiaEval Group 7)",
        "variant": args.variant,
        "model": {
            "name": model_name,
            "revision_requested": revision,
            "revision_resolved": resolve_model_revision(model_name, revision),
            "num_labels": mcfg["num_labels"],
        },
        "data": {
            "root": str(resolve_path(data_root)),
            "text_column": cfg["data"]["text_column"],
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
            "leakage_check": leakage,
            "english_column_used": False,
        },
        "tokenizer": {"max_length": max_length, "padding": tokcfg["padding"], "truncation": True},
        "training": {
            "seed": args.seed,
            "epochs": epochs,
            "learning_rate": lr,
            "train_batch_size": train_bs,
            "eval_batch_size": eval_bs,
            "gradient_accumulation_steps": grad_accum,
            "weight_decay": tcfg["weight_decay"],
            "warmup_ratio": tcfg["warmup_ratio"],
            "lr_scheduler_type": tcfg["lr_scheduler_type"],
            "precision": pick_precision(precision),
            "selection_metric": tcfg["metric_for_best_model"],
            "unsupported_training_args_dropped": dropped,
            "wall_clock_seconds": round(elapsed, 1),
        },
        "selection": {
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_macro_f1": trainer.state.best_metric,
        },
        "environment": environment_fingerprint(),
    }
    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "validation_history.json", history)
    write_json(out_dir / "validation_final.json", final_val)

    print(json.dumps(
        {
            "variant": args.variant,
            "best_validation_macro_f1": trainer.state.best_metric,
            "model_dir": str(out_dir),
            "minutes": round(elapsed / 60, 1),
        },
        indent=2,
    ))
    return out_dir


if __name__ == "__main__":
    main()
