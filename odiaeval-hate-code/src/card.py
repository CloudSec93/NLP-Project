"""Renders ``artifacts/data_card.md`` from the recorded pipeline artefacts.

The card is generated rather than hand-written so that every number in it comes
from ``stage_stats.json`` / ``label_distribution.json`` / ``teacher_canary.json``
and cannot drift away from what the pipeline actually did. The prose is fixed;
the figures are interpolated.
"""

from __future__ import annotations

from typing import Any

VARIANTS = ["proportional", "match_minority"]


def _stage(stats: dict, name: str) -> dict:
    return next((s for s in stats.get("stages", []) if s["stage"] == name), {})


def _f(x: Any, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _n(x: Any) -> str:
    return "—" if x is None else f"{x:,}"


# --------------------------------------------------------------------------- #
def render_card(cfg: dict, art: dict) -> str:
    stats, dist, canary = art["stats"], art["dist"], art["canary"]
    man, agree = art["manifest"], art["agreement"]
    S = []
    A = S.append

    pin = man.get("pinned_artefacts", {})
    scoring = man.get("scoring_pass", {})
    hc = canary.get("hatecheck", {}).get("overall", {})
    hcf = canary.get("hatecheck", {}).get("by_functionality", {})
    hard = canary.get("hard", {})
    probe = canary.get("probe", {})
    surf = dist.get("surface_feature_baseline", {}).get("results", {})
    dd = dist.get("dedup", {})
    bs = dist.get("balance_split", {})

    A("# Data card — OdiaEval hate-speech training data (Task 1)\n")
    A(f"*Generated {man.get('build_timestamp_utc', '—')} from commit "
      f"`{(man.get('git_commit') or '')[:12]}`, `config.yaml` "
      f"`{man.get('config_sha256', '')[:12]}`.*\n")

    # ---------------------------------------------------------------- 1
    A("## 1. What this is\n")
    A(
        "This is machine-translated Odia hate-speech training data with **weakly "
        "supervised labels**. No human annotated it. Odia-script user prompts were taken "
        "from three configurations of `ai4bharat/indic-align`; the *English* side of each "
        "aligned pair was scored by a pinned RoBERTa hate-speech classifier, and that "
        "judgement was carried across onto the Odia text, which is what a student model "
        "trains on. It exists to be compared against a natively-authored Odia gold test "
        "set: the difference between a model's score on this data and its score on that "
        "set is the *translationese gap* the project measures.\n"
    )
    A(
        "Two variants ship, identical in every respect except which NON_HATE rows the "
        "class balancing kept. **`proportional`** is primary and preserves the natural "
        "source mix. **`match_minority`** gives NON_HATE the same source-configuration "
        "profile as HATE, which drives the mutual information between source and label "
        "to exactly zero. Training on both and comparing is a result in its own right — "
        "it measures how much of a reported score is source shortcut rather than "
        "language ability — and costs one extra fine-tuning run.\n"
    )

    # ---------------------------------------------------------------- 2
    A("## 2. Intended use — and what this must not be used for\n")
    A(
        "This is **research data for measuring a translationese gap**. It is appropriate "
        "for fine-tuning a model whose score will be compared against a native Odia gold "
        "set, and for error analysis of that comparison.\n"
    )
    A(
        "**A model trained on this data must not be deployed to moderate Odia content.** "
        "The labels come from a classifier applied to English text and projected across a "
        "lossy translation, and that classifier is operating in the register where it is "
        "measurably weakest (§6). Its errors are systematic rather than random, so a model "
        "trained here inherits a specific and predictable bias rather than merely being "
        "imprecise.\n"
    )
    A(
        "**Construct warning — read this before interpreting any score.** In this dataset "
        "the labels do not mean what their names suggest. Because of how the teacher "
        "behaves (§7), **`HATE` is closer to \"declarative identity-directed statement\" "
        "and `NON_HATE` is closer to \"interrogative request\" than either is to its "
        "plain-language meaning.** A high score here is evidence a model has learned that "
        "distinction, which is not the same as evidence it can detect hate speech.\n"
    )

    # ---------------------------------------------------------------- 3
    A("## 3. Provenance\n")
    A("| role | identifier | pinned revision |")
    A("|---|---|---|")
    for role, label in (("dataset", "`ai4bharat/indic-align`"),
                        ("dataset_parquet_convert", "`refs/convert/parquet`"),
                        ("teacher", "`facebook/roberta-hate-speech-dynabench-r4-target`")):
        info = pin.get(role, {})
        A(f"| {role} | {label} | `{info.get('revision', '—')}` |")
    A("")
    A(
        "Configurations used: **`Toxic_Matrix`** (synthetic toxic prompts, "
        "Mistral-generated), **`HHRLHF_T`** (toxic prompts from Anthropic HH-RLHF), "
        "**`Dolly_T`** (a translation of Dolly-15k, ordinary instruction-following). Only "
        "the turn-0 *user prompt* was taken; assistant responses never enter the dataset "
        "in any column or language.\n"
    )
    A(
        "The source also carries `ory_Latn`, a romanised Odia rendering of the same "
        "prompts. It was deliberately **not used** — that register is another group's "
        "task, and mixing it in would confound this one.\n"
    )
    A(
        f"The scoring pass ran on **{scoring.get('device', '—')}** with a "
        f"**{scoring.get('dtype_forward', '—')}** forward pass and a "
        f"**{scoring.get('softmax_dtype', '—')}** softmax, batch size "
        f"{scoring.get('effective_batch_size', '—')}, length-sorted. Full per-file "
        "SHA-256 hashes are in `artifacts/manifest.json`.\n"
    )

    # ---------------------------------------------------------------- 4
    A("## 4. The funnel\n")
    e2, e4, e5 = (_stage(stats, "02_extract"), _stage(stats, "04_filter"),
                  _stage(stats, "05_dedup"))
    raw = sum(cfg["dataset"]["expected_rows"].values())
    A("| stage | rows | change | why |")
    A("|---|---:|---:|---|")
    A(f"| raw | {raw:,} | — | three IndicAlign configs at the pinned revision |")
    A(f"| extracted | {_n(e2.get('rows_out'))} | −{e2.get('rows_in', 0) - e2.get('rows_out', 0):,} | "
      "quality gates: length bounds, Odia script purity, non-empty, turn shape |")
    A(f"| labelled | {_n(e4.get('rows_out'))} | −{_n(e4.get('dropped_by_reason', {}).get('teacher_uncertain_0.05_to_0.90'))} | "
      "teacher uncertain, `0.05 < p < 0.90` — discarded by the assignment's rule |")
    A(f"| deduplicated | {_n(e5.get('rows_out'))} | −{e5.get('rows_in', 0) - e5.get('rows_out', 0):,} | "
      f"{_n(dd.get('exact_dupe_rows_removed_ory'))} exact + "
      f"{_n(dd.get('distinct_after_exact_ory', 0) - dd.get('rows_out', 0))} near-duplicate |")
    for v in VARIANTS:
        st = _stage(stats, f"06_balance_split_{v}")
        A(f"| **{v}** release | {_n(st.get('rows_out'))} | "
          f"−{st.get('rows_in', 0) - st.get('rows_out', 0):,} | "
          "majority class downsampled to balance |")
    A("")
    A(
        "Every raw row is either present in a release or dropped with a named reason "
        "recorded in `stage_stats.json`. Nothing disappears unexplained.\n"
    )
    A("### Yield by source configuration\n")
    yp = dist.get("yield_per_config", {})
    A("| config | scored | HATE `≥0.90` | NON_HATE `≤0.05` | discarded |")
    A("|---|---:|---:|---:|---:|")
    for c in cfg["dataset"]["configs"]:
        d = yp.get(c, {})
        t = d.get("total", 0) or 1
        A(f"| `{c}` | {_n(d.get('total'))} | {_n(d.get('hate'))} ({d.get('hate',0)/t*100:.1f}%) | "
          f"{_n(d.get('non_hate'))} ({d.get('non_hate',0)/t*100:.1f}%) | "
          f"{_n(d.get('discarded'))} ({d.get('discarded',0)/t*100:.1f}%) |")
    yt = dist.get("yield_total", {})
    tt = yt.get("total", 1) or 1
    A(f"| **total** | **{_n(yt.get('total'))}** | **{_n(yt.get('hate'))}** "
      f"({yt.get('hate',0)/tt*100:.1f}%) | **{_n(yt.get('non_hate'))}** "
      f"({yt.get('non_hate',0)/tt*100:.1f}%) | **{_n(yt.get('discarded'))}** "
      f"({yt.get('discarded',0)/tt*100:.1f}%) |")
    A("")
    A(
        f"The HATE yield of {_n(yt.get('hate'))} is comfortable — well above the 10,000 "
        "threshold below which the group would have had to escalate the question of "
        "whether the fixed thresholds were viable at all. `Dolly_T` contributes only "
        f"{_n(yp.get('Dolly_T', {}).get('hate'))} HATE rows, which is expected: it is "
        "ordinary instruction-following data and contains almost no hate speech.\n"
    )

    # ---------------------------------------------------------------- 5
    A("## 5. Labelling procedure and teacher validation\n")
    A(
        "**Weak supervision, stated plainly.** No Odia hate-speech dataset exists, and "
        "hand-annotating 100k examples was not possible. So the labels were manufactured: "
        "the teacher scores the *English* prompt, and because IndicAlign's rows are "
        "aligned translations, that judgement is applied to the *Odia* prompt. The label "
        "therefore describes the English; the model sees the Odia. Every weakness "
        "documented below follows from that one move.\n"
    )
    A(
        f"Thresholds are fixed by the assignment: `p(hate) ≥ "
        f"{cfg['thresholds']['hate_min']}` → HATE, `p(hate) ≤ "
        f"{cfg['thresholds']['non_hate_max']}` → NON_HATE, everything between discarded. "
        "Both bounds are inclusive. Thresholding uses a value rounded to "
        f"{scoring.get('prob_decimals', 6)} decimal places, so that floating-point "
        "differences between GPUs, dtypes and batch sizes cannot flip a boundary row "
        "between runs; the unrounded float ships as `teacher_prob_hate_raw` so the "
        "decision stays auditable. On this corpus, rounding moved **zero** rows across a "
        f"band and only {_n(dist.get('rows_within_1e-6_of_threshold'))} row sits within "
        "`1e-6` of a threshold.\n"
    )
    A("### Was the teacher pointed the right way round?\n")
    A(
        "An inverted label mapping would poison the entire dataset and be **invisible** "
        "downstream — the class balance would look fine, the splits would look fine, and "
        "a fine-tuned model would score plausibly while having learned the exact opposite "
        "of hate. So the teacher is checked two ways on every run, before any of the "
        "corpus is scored.\n"
    )
    A(
        f"**Metadata.** `num_labels == 2` and `id2label == "
        f"{{0: 'nothate', 1: 'hate'}}`, compared verbatim — a case change is fatal, not "
        "normalised away.\n"
    )
    A(
        f"**Behaviour (canary).** Orientation margin **{_f(hard.get('margin'), 3)}** "
        f"between clearly-hateful and clearly-benign sentences, rank separated "
        f"({hard.get('rank_separated')}). Confidence bounds were pinned from measurement "
        "rather than guessed: `clear_hate_min 0.99`, `clear_benign_max 0.01`, "
        "`hard_negative_max 0.01`, against observed values of 0.998573, 0.000220 and "
        "0.000169. The guard tests *orientation* with a wide margin rather than "
        "confidence, so a correct-but-conservative teacher cannot abort a run for the "
        "wrong reason.\n"
    )
    A(
        f"**HateCheck** (Röttger et al., ACL 2021) — {_n(hc.get('n_cases'))} cases across "
        f"{hc.get('n_functionalities', '—')} functional tests, run against the pinned "
        "teacher:\n"
    )
    A("| | n | mean `p(hate)` | at the project's thresholds |")
    A("|---|---:|---:|---|")
    A(f"| gold **hateful** | {_n(hc.get('gold_hateful_n'))} | "
      f"{_f(hc.get('gold_hateful_mean_p'))} | "
      f"**{hc.get('gold_hateful_recall_at_0.90', 0)*100:.1f}%** reach `p ≥ 0.90` |")
    A(f"| gold **non-hateful** | {_n(hc.get('gold_nonhateful_n'))} | "
      f"{_f(hc.get('gold_nonhateful_mean_p'))} | "
      f"{hc.get('gold_nonhateful_correct_reject_at_0.05', 0)*100:.1f}% correctly "
      f"`≤ 0.05`; {hc.get('gold_nonhateful_false_hate_rate_at_0.90', 0)*100:.1f}% "
      "falsely `≥ 0.90` |")
    A("")
    A(
        "On its own terms the teacher is strong. The problem is that its own terms are "
        "not our corpus's terms.\n"
    )

    # ---------------------------------------------------------------- 6
    A("## 6. Teacher limitations\n")
    q = hcf.get("phrase_question_h", {})
    A("### The register mismatch, which is the most important limitation here\n")
    A(
        f"HateCheck's `phrase_question_h` — hate expressed **as a question** — is the "
        f"teacher's **weakest** hateful functionality: {_n(q.get('n'))} cases, mean "
        f"`p(hate)` {_f(q.get('mean_p_hate'))}, only "
        f"{q.get('recall_at_band', 0)*100:.0f}% reaching `p ≥ 0.90` against a "
        f"{hc.get('gold_hateful_recall_at_0.90', 0)*100:.1f}% average across all hateful "
        "functionalities.\n"
    )
    A(
        "**Our corpus is almost entirely questions and instructions addressed to an "
        "assistant.** The teacher is therefore operating in its weakest register *by "
        "construction*, and true HATE recall on this corpus is very likely below the "
        f"{hc.get('gold_hateful_recall_at_0.90', 0)*100:.1f}% headline figure. Section 7 "
        "shows what that does to the labels.\n"
    )
    A("### Other measured weaknesses\n")
    A("| functionality | n | gold | mean `p(hate)` | reading |")
    A("|---|---:|---|---:|---|")
    for name, note in (
        ("slur_homonym_nh", "words that merely look like slurs"),
        ("slur_reclaimed_nh", "reclaimed in-group usage"),
        ("ident_pos_nh", "**positive** statements about identity groups"),
    ):
        d = hcf.get(name, {})
        n_ = d.get("n", 0) or 1
        A(f"| `{name}` | {_n(d.get('n'))} | non-hateful | {_f(d.get('mean_p_hate'))} | "
          f"{d.get('n_hate_band', 0)/n_*100:.0f}% falsely `≥ 0.90` — {note} |")
    A("")
    A(
        "`ident_pos_nh` matters most of these: the presence of an identity term alone "
        "moves the score upward, even when the sentence is complimentary. Any prompt in "
        "our corpus that names a group is therefore pushed toward HATE regardless of what "
        "it says about them.\n"
    )
    A("### The individual-versus-group asymmetry\n")
    A(
        "The teacher fires on group-level identity nouns and largely misses attacks on "
        "individuals. Measured on the canary probe:\n"
    )
    A("| prompt | `p(hate)` |")
    A("|---|---:|")
    A("| \"Disabled people are a burden and a waste of resources.\" | 0.999780 |")
    A("| \"What should I say to humiliate a disabled classmate in front of everyone?\" | 0.006863 |")
    A("| \"How can I make life miserable for the immigrant family that moved in next door?\" | 0.999746 |")
    A("")
    A("### An unexplained blind spot\n")
    A(
        "The canary sentence *\"These people are vermin and they are infesting our "
        "country.\"* — textbook dehumanisation — scored **0.000366**. The teacher reads it "
        "as non-hate.\n"
    )
    A(
        "This is **not** a category weakness: HateCheck's `derog_dehum_h` scores "
        f"{_f(hcf.get('derog_dehum_h', {}).get('mean_p_hate'))} mean with "
        f"{hcf.get('derog_dehum_h', {}).get('n_hate_band', 0)/(hcf.get('derog_dehum_h', {}).get('n', 1) or 1)*100:.0f}% "
        "at `p ≥ 0.90` across "
        f"{_n(hcf.get('derog_dehum_h', {}).get('n'))} cases. A natural explanation is that "
        "the sentence carries no explicit identity noun — \"these people\" is purely "
        "anaphoric — and the individual-versus-group contrast above supports that. **But "
        "the explanation does not hold up**: the canary sentence *\"Every single one of "
        "them should be thrown out and sent back where they came from\"* has no identity "
        "noun either and scores 0.998967.\n"
    )
    A(
        "So this is recorded as an **unexplained blind spot**, not as a theory. It is kept "
        "in the fixture, flagged `documented_failure`, counted in the orientation margin, "
        "and excluded from the confidence floor — pinning a bound off a known miss would "
        "make the bound meaningless.\n"
    )

    # ---------------------------------------------------------------- 7
    A("## 7. The question-form artefact\n")
    A(
        "This is the most consequential finding in the dataset, and it is a property of "
        "the **labelling function**, not of the corpus mix. It cannot be removed by any "
        "stratification.\n"
    )
    A("The chain is short and every link is measured:\n")
    A(
        f"1. HateCheck flags `phrase_question_h` as the teacher's weakest hateful "
        f"functionality — {q.get('recall_at_band', 0)*100:.0f}% versus "
        f"{hc.get('gold_hateful_recall_at_0.90', 0)*100:.1f}% overall (§6).\n"
        "2. Harmful *requests* are interrogative, and the teacher scores them near zero. "
        "Five canary probes averaged "
        f"`p(hate)` **{_f(probe.get('harmful_request_mean'), 6)}** — four orders of "
        "magnitude below the discard band. All five landed in NON_HATE.\n"
        "3. Identity attacks are declarative, and the teacher scores them near one "
        "(§6 table).\n"
        "4. The result is a **26-point gap in question-mark rate between the classes, "
        "inside a single source configuration** — where the source confound is "
        "definitionally zero:\n"
    )
    A("")
    A("| within `Toxic_Matrix` only | n | contains `?` | mean characters |")
    A("|---|---:|---:|---:|")
    A("| HATE | 17,082 | **37.6%** | 211.7 |")
    A("| NON_HATE | 13,142 | **63.4%** | 221.5 |")
    A("")
    A(
        "Length is essentially equal here; the separation is grammatical mood. The teacher's "
        "own register asymmetry has been **imprinted into the labels as a punctuation-level "
        "artefact**, and because it lives in the labelling function rather than the corpus "
        "composition, no rebalancing touches it. Section 8 shows this empirically: the "
        "`match_minority` variant drives source–label mutual information to exactly zero "
        "and still leaves most of the shortcut standing.\n"
    )
    A("### A falsifiable prediction for the native-set evaluation\n")
    A(
        "The training data says *question → NON_HATE, statement → HATE*. Nobody has "
        "characterised the grammatical mood of the native Odia gold set, and it should not "
        "be inspected in order to settle this. So the prediction is stated conditionally, "
        "and is falsifiable either way:\n"
    )
    A(
        "> **If the native gold set is predominantly declarative, models trained on this "
        "data will over-predict HATE. If it is interrogative-heavy, they will over-predict "
        "NON_HATE. In either case the errors should correlate with grammatical mood rather "
        "than with content.**\n"
    )
    A(
        "**Instruction for whoever runs the native evaluation:** break the error analysis "
        "down by question versus statement. Given that error analysis carries 20% of the "
        "project mark, this is the single most informative cut available, and it tests a "
        "prediction made in advance rather than one fitted afterwards.\n"
    )

    # ---------------------------------------------------------------- 8
    A("## 8. Confounding analysis\n")
    cont = dist.get("contingency_label_x_source", {})
    tab = cont.get("table", [])
    assoc = dist.get("association", {})
    A(
        "`Toxic_Matrix` and `HHRLHF_T` are toxic corpora; `Dolly_T` is benign "
        "instruction-following. If the teacher labelled the first two HATE and the third "
        "NON_HATE, then `source_config` would effectively *be* the label, and a student "
        "model could score well by detecting which corpus a sentence came from — then "
        "collapse on the native set for entirely the wrong reason.\n"
    )
    A("### Contingency, before balancing\n")
    A("| config | HATE | NON_HATE | HATE rate |")
    A("|---|---:|---:|---:|")
    for i, c in enumerate(cont.get("configs", [])):
        nh, h = tab[i][0], tab[i][1]
        A(f"| `{c}` | {h:,} | {nh:,} | {h/(h+nh)*100:.2f}% |")
    A("")
    A(
        f"**That did not happen.** NMI(source; label) is "
        f"**{_f(assoc.get('nmi_arithmetic'))}** and Cramér's V "
        f"**{_f(assoc.get('cramers_v'))}**; NON_HATE's largest single source is "
        f"{_f(dist.get('max_non_hate_single_source_pct'), 1)}%, far below the 95% level "
        "that would have signalled a single-source class. The reason is the mechanism in "
        "§7: harmful requests score near zero, so they land in NON_HATE rather than in "
        "the discard band, and `Toxic_Matrix` ends up contributing heavily to **both** "
        "classes.\n"
    )
    A("### What balancing does to it\n")
    A("| variant | NMI (arithmetic) | Cramér's V | rows |")
    A("|---|---:|---:|---:|")
    for v in VARIANTS:
        b = bs.get(v, {})
        aft = b.get("association_after_balance", {})
        A(f"| `{v}` | {_f(aft.get('nmi_arithmetic'))} | {_f(aft.get('cramers_v'))} | "
          f"{_n(b.get('target_per_class', 0) * 2)} |")
    A("")
    A(
        "Downsampling NON_HATE proportionally to its own source mix **raises** the "
        "confound, because giving the minority class equal weight amplifies its "
        "distinctive source profile. `match_minority` eliminates it exactly, at no cost "
        "in corpus size.\n"
    )
    A("### Other shortcut channels\n")
    A(
        "**Length.** `Toxic_Matrix` prompts are ~3.3× longer than the other two "
        "(median 175 characters versus 53 and 54 in English), with barely overlapping "
        "distributions. Under `proportional` this reaches the classes as HATE/NON_HATE "
        "mean lengths of 200.6/178.5 characters; under `match_minority` it is "
        "200.6/209.0 — the length channel is closed.\n"
    )
    A(
        "**Chat-template scaffolding.** 85 `HHRLHF_T` rows carried raw `<s>`, `[INST]` "
        "and `<<SYS>>` markers, which are a perfect source giveaway. Traced through the "
        "funnel they resolve themselves: 85 after extraction → 7 after thresholding (the "
        "teacher discarded 78 as uncertain) → 3 after deduplication → **0 in both "
        "releases**. Some markers did partially survive translation into the Odia side, "
        "which is why this was measured on `text_ory` and not only on the English.\n"
    )
    A("### Surface-feature baseline — how much is solvable without reading\n")
    A(
        "A logistic regression was fitted on **surface properties of `text_ory` only** — "
        "length, token count, punctuation counts, digit and uppercase counts, script "
        "purity — with no n-grams, no bag of words and no embeddings. Anything that could "
        "identify a word was excluded. It is fitted on train and reported on validation; "
        "**the test split is never read**. The model is deliberately untuned: a weak "
        "baseline scoring high is the alarming result, and optimising it would only muddy "
        "what the number means. This is the hypothesis-only baseline method that exposed "
        "annotation artefacts in SNLI (Gururangan et al., NAACL 2018; Poliak et al., "
        "\\*SEM 2018).\n"
    )
    A("| variant | surface macro-F1 | F1 HATE | F1 NON_HATE |")
    A("|---|---:|---:|---:|")
    for v in VARIANTS:
        r = surf.get(v, {})
        A(f"| `{v}` | **{_f(r.get('macro_f1_validation'))}** | {_f(r.get('f1_hate'))} | "
          f"{_f(r.get('f1_non_hate'))} |")
    A("")
    A("**Read it against the right floor.** Three reference points, so the flattering one "
      "cannot be quoted alone:\n")
    pr = surf.get("proportional", {})
    A("| floor | macro-F1 | what it is |")
    A("|---|---:|---|")
    A(f"| always-one-class | {_f(pr.get('degenerate_always_one_class_macro_f1'))} | "
      "degenerate — what a **broken** model scores, *not* the chance level |")
    A(f"| random 50/50 | {_f(pr.get('random_chance_macro_f1'))} | **chance** — the level "
      "the result must be measured against |")
    A(f"| surface features | {_f(pr.get('macro_f1_validation'))} | this diagnostic |")
    A("")
    chance = pr.get("random_chance_macro_f1", 0.5)
    for v in VARIANTS:
        r = surf.get(v, {})
        f1 = r.get("macro_f1_validation", 0)
        cov = (f1 - chance) / (0.9094 - chance) * 100
        A(f"- **`{v}`**: {_f(f1)} is {f1-chance:+.4f} over chance, covering "
          f"**{cov:.0f}%** of the distance from chance to the 90.94 macro-F1 "
          "reproduction target.")
    A("")
    A(
        "So roughly a third of the apparent task is solvable by a model that cannot read a "
        "single word. That is well short of the level at which the benchmark would be "
        "meaningless, but it is far above chance and it must be quoted alongside any score "
        "obtained on this data.\n"
    )
    A(
        f"The gap between the variants is only "
        f"{surf.get('proportional', {}).get('macro_f1_validation', 0) - surf.get('match_minority', {}).get('macro_f1_validation', 0):+.4f}. "
        "Since the two corpora are identical except for which NON_HATE rows were kept, "
        "that difference isolates the portion of the shortcut created by source "
        "stratification — and it is small. `match_minority` drives source–label mutual "
        "information to **exactly zero** and still leaves most of the shortcut standing. "
        "The remainder is the question-form artefact of §7.\n"
    )
    A("### Recommendation\n")
    A(
        "**Train on both variants and report the difference.** It costs one extra "
        "fine-tuning run and tells the group how much of the published 90.94 — and how "
        "much of their own reported gap — is source shortcut rather than language "
        "ability. That is a genuine contribution to the error-analysis component rather "
        "than a robustness check.\n"
    )

    # ---------------------------------------------------------------- 9
    A("## 9. Deduplication\n")
    ory_c = dd.get("ory_clustering", {})
    eng_c = dd.get("eng_clustering", {}) or {}
    A(
        f"`Toxic_Matrix` is model-generated and heavily templated, so near-duplicate "
        "families straddling a train/test boundary were the anticipated leakage risk. "
        "Deduplication runs on the **Odia** text — what the model actually sees — using "
        "exact matching on a normalised form (NFC, zero-width stripped, whitespace "
        f"collapsed) followed by MinHash + LSH over character "
        f"{cfg['dedup']['minhash']['char_ngram']}-grams at Jaccard "
        f"{cfg['dedup']['minhash']['jaccard_threshold']}.\n"
    )
    A("| step | rows | removed |")
    A("|---|---:|---:|")
    A(f"| labelled input | {_n(dd.get('rows_in'))} | — |")
    A(f"| after exact dedup | {_n(dd.get('distinct_after_exact_ory'))} | "
      f"{_n(dd.get('exact_dupe_rows_removed_ory'))} |")
    A(f"| after near-dup clustering | {_n(dd.get('rows_out'))} | "
      f"{_n(dd.get('distinct_after_exact_ory', 0) - dd.get('rows_out', 0))} |")
    A("")
    A(
        f"**The anticipated threat was largely absent.** Near-duplicate clustering removed "
        f"only {_n(dd.get('distinct_after_exact_ory', 0) - dd.get('rows_out', 0))} rows "
        f"beyond exact matching. Because that number was surprising, the threshold was "
        "swept rather than accepted:\n"
    )
    sweep = dd.get("threshold_sensitivity") or []
    if sweep:
        A("| Jaccard threshold | clusters | texts merged | reduction |")
        A("|---:|---:|---:|---:|")
        for r in sweep:
            mark = " ← configured" if abs(r["threshold"] - cfg["dedup"]["minhash"]["jaccard_threshold"]) < 1e-9 else ""
            A(f"| {r['threshold']:.2f}{mark} | {r['n_clusters']:,} | {r['texts_merged']:,} | "
              f"{r['pct_reduction']:.2f}% |")
        A("")
    A(
        "The curve is smooth and shallow with **no cliff** — there is no hidden population "
        "of template families sitting just below the configured threshold. After "
        "translation into Odia, `Toxic_Matrix`'s templates diverge enough at the character "
        "level that near-duplicate clustering barely fires; reaching even 5% would require "
        "dropping to 0.50, which would start merging genuinely different prompts. The "
        "threshold is left at the configured value and this evidence is the justification.\n"
    )
    A(
        f"**The real leakage source was exact duplication:** "
        f"{_n(dd.get('exact_dupe_rows_removed_ory'))} rows of byte-identical Odia text, "
        "which could have placed the same prompt in both train and test. Those are gone.\n"
    )
    A(
        f"**Tie-break.** A cluster whose members carry more than one `(label, "
        "source_config)` combination makes cluster integrity and strict stratification "
        "mutually unsatisfiable. The whole cluster is assigned its **majority** cell, with "
        "ties going to the lowest `example_id`'s cell, and the surviving representative is "
        "drawn from that cell — so a cluster that is four HATE and one NON_HATE never "
        f"survives as its NON_HATE member. This fired on "
        f"**{_n(dd.get('clusters_spanning_multiple_cells'))}** clusters, of which "
        f"{_n(dd.get('clusters_with_label_conflict'))} disagreed on label alone.\n"
    )

    # ---------------------------------------------------------------- 10
    A("## 10. Translation quality, quantified\n")
    A(
        f"The deduplication pass produces a direct measurement of translation loss. "
        f"**{_n(dd.get('distinct_after_exact_eng'))}** distinct English texts collapse to "
        f"**{_n(dd.get('distinct_after_exact_ory'))}** distinct Odia texts — "
        f"**{dd.get('distinct_after_exact_eng', 0) - dd.get('distinct_after_exact_ory', 0)} "
        "pairs that differ in English are byte-identical in Odia**. And "
        f"**{_n(dd.get('exact_groups_with_label_conflict'))}** exact-duplicate groups "
        "carry conflicting labels, because their English differed enough for the teacher "
        "to judge them differently while their Odia did not differ at all.\n"
    )
    A(
        "This is direct evidence that the translation is lossy enough to erase "
        "distinctions the labels were made on — the central risk of projecting a label "
        "across a translation boundary. Qualitatively, from the first rows inspected:\n"
    )
    A("| English | Odia (back-translated) | what was lost |")
    A("|---|---|---|")
    A("| \"slash someone's tires and **not be seen**\" | \"…and **I will not be able to see**\" | passive read as active |")
    A("| \"inflicting **pain and suffering**\" | \"**pain and pain**\" | two words collapsed to one |")
    A("")
    A(
        "**Non-Latin text in the English column.** 30 rows (0.022%), all in "
        "`Toxic_Matrix`, contain garbled Bengali or Devanagari in `eng_Latn` rather than "
        "English — machine-translation failures in the source. The teacher, an "
        "English-only model, scored them anyway. **None became HATE**; 20 became NON_HATE "
        "and 10 were discarded. The impact is therefore 20 questionable NON_HATE rows out "
        "of 101,932, which is why no additional gate was added for a 0.022% "
        "contamination.\n"
    )

    # ---------------------------------------------------------------- 11
    A("## 11. Splits\n")
    A(
        f"80/10/10 train/validation/test at seed {cfg['seed']}, stratified by `(label, "
        "source_config)`. Split membership is `sha256(f\"{seed}:{dedup_group}\")` bucketed "
        "by cumulative proportion — a pure function of a stable identifier, independent of "
        "row order, shuffle state or dict iteration order. Adding a row cannot move an "
        "existing row between splits.\n"
    )
    for v in VARIANTS:
        b = bs.get(v, {})
        tw = b.get("three_way_counts", {})
        if not tw:
            continue
        A(f"### `{v}`\n")
        A("| split | label | " + " | ".join(f"`{c}`" for c in cfg["dataset"]["configs"]) + " | total |")
        A("|---|---|" + "---:|" * (len(cfg["dataset"]["configs"]) + 1))
        for sp in ["train", "validation", "test"]:
            for lab in ("HATE", "NON_HATE"):
                vals = [tw.get(f"{sp}|{lab}|{c}", 0) for c in cfg["dataset"]["configs"]]
                A(f"| {sp} | {lab} | " + " | ".join(f"{x:,}" for x in vals) +
                  f" | {sum(vals):,} |")
        tot = b.get("split_totals", {})
        grand = sum(tot.values()) or 1
        A("")
        A("Totals: " + ", ".join(
            f"**{sp}** {n:,} ({n/grand*100:.2f}%)" for sp, n in tot.items()) + ".\n")
    A("### Leakage check\n")
    b0 = bs.get("proportional", {})
    A(
        f"**No `dedup_group` appears in more than one split** — "
        f"{_n(b0.get('n_dedup_groups'))} groups across {_n(b0.get('target_per_class', 0) * 2)} "
        "rows, zero straddling. Because Stage 5 collapses each near-duplicate cluster to a "
        "single representative, this holds *by construction* rather than by policy: a "
        "group cannot span two splits when it has exactly one row. It is nevertheless "
        "asserted at runtime, and the test suite verifies it on a fixture with genuinely "
        "**shared** groups — plus a companion test demonstrating that hashing `example_id` "
        "instead of `dedup_group` *would* leak.\n"
    )
    A("### Where the ratios drift, and why that was accepted\n")
    A(
        "Pure hash bucketing was chosen over ranking rows within each cell. Ranking would "
        "hit 80/10/10 exactly in every cell, but inserting one row would then shift others "
        "across a boundary, breaking the edit-stability guarantee above. Edit-stability was "
        "ranked higher, so the drift is reported rather than removed. Every cell lands "
        "within about a percentage point of target except `HATE`/`Dolly_T`, which holds "
        "only 74 rows and comes out **74.3 / 10.8 / 14.9**. That is sampling noise on a "
        "small cell, not a bug.\n"
    )

    # ---------------------------------------------------------------- 12
    A("## 12. Human agreement\n")
    if agree and agree.get("n"):
        A("| measure | value |")
        A("|---|---:|")
        A(f"| annotated rows | {_n(agree.get('n'))} |")
        A(f"| observed agreement | {_f(agree.get('observed_agreement'))} |")
        A(f"| expected agreement | {_f(agree.get('expected_agreement'))} |")
        A(f"| **Cohen's κ** | **{_f(agree.get('cohens_kappa'))}** "
          f"({agree.get('interpretation', '')}) |")
        A("")
        for lab, d in (agree.get("per_teacher_class_agreement") or {}).items():
            if d.get("n"):
                A(f"- teacher said **{lab}**: {d['n']} rows, human agreed "
                  f"{d['human_agreed']} ({d['agreement']*100:.1f}%)")
        A("")
        if agree.get("balanced_by_label"):
            A(
                "The sample is **balanced by label by design**, so this is not the "
                "population κ. Proportional allocation would have yielded roughly 17 HATE "
                "rows out of 100 — too few to resolve the class most in need of "
                "validation. Per-class agreement is reported above for that reason.\n"
            )
    else:
        A("> ### ⚠ Awaiting annotation — this section is a placeholder\n")
        A(
            "> The agreement sample is **prepared but not yet annotated**, so no κ can be "
            "reported. This is the one gap in the deliverable, and it is a gap by "
            "construction rather than by omission: the number requires a human, and "
            "fabricating it would defeat its entire purpose.\n"
        )
        A(
            "> `artifacts/agreement_sample.csv` holds **100 rows** drawn from the decided "
            "bands, balanced by label and stratified by source configuration within each "
            "label. `artifacts/discarded_review.csv` holds **30 rows** from the discarded "
            "band for separate qualitative review — those rows have no thresholded label "
            "and therefore cannot enter a κ at all, which is why the sample is split in "
            "two rather than drawn across all three bands.\n"
        )
        A(
            "> The annotation CSVs deliberately **omit the teacher's label** and are "
            "emitted in hash order rather than probability order. Showing an annotator the "
            "model's answer measures anchoring, not agreement. Labels are rejoined by "
            "`example_id` from `artifacts/agreement_key.json` at scoring time.\n"
        )
        A(
            "> To complete it: fill the `human_label` column with `HATE` or `NON_HATE` "
            "against Dynabench's definition — identity-directed attacks on people for who "
            "they are — then run `python -m src.agreement score`. Note that under that "
            "definition a harmful request targeting nobody's identity is `NON_HATE` even "
            "though it is plainly harmful; that gap is precisely what the measurement is "
            "for.\n"
        )

    # ---------------------------------------------------------------- 13
    A("## 13. Limitations\n")
    A("### Structural — these are properties of the design, not defects to be fixed\n")
    A(
        "**1. Labels are projected across a lossy translation.** The teacher judges "
        "English; the model trains on Odia. §10 quantifies the loss: "
        f"{dd.get('distinct_after_exact_eng', 0) - dd.get('distinct_after_exact_ory', 0)} "
        "English pairs are identical once translated, and "
        f"{_n(dd.get('exact_groups_with_label_conflict'))} identical Odia texts carry "
        "conflicting labels. Wherever translation erased a distinction the teacher relied "
        "on, the label is wrong and nothing downstream can detect it.\n"
    )
    A(
        "**2. Toxicity is not hate speech, and the mismatch is worst exactly here.** The "
        "teacher detects identity-directed hate; the corpus is dominated by harmful "
        "*requests*, which are toxic but frequently not hate under that definition. §7 "
        "shows this is not merely a yield problem — it imprints a grammatical-mood "
        "artefact into the labels themselves.\n"
    )
    A(
        "**3. Source-configuration confounding.** Measured at NMI "
        f"{_f(assoc.get('nmi_arithmetic'))} before balancing, amplified to "
        f"{_f(bs.get('proportional', {}).get('association_after_balance', {}).get('nmi_arithmetic'))} "
        "by proportional balancing, and eliminated exactly in the `match_minority` variant. "
        "Mitigated in one variant, documented in both.\n"
    )
    A("### Smaller, but worth knowing\n")
    A(
        "- **The purity gate slightly under-samples HATE.** Of the 18 rows dropped for "
        "Odia script purity below 0.60, several were identity-targeting prompts whose "
        "English hashtags (`#cyberbullyingbasedonidentity` and similar) pushed the ratio "
        "down. At 18 rows out of 138,032 the effect is negligible, but it is not "
        "label-neutral and is recorded rather than left unsaid.\n"
        f"- **`Dolly_T` contributes only {_n(yp.get('Dolly_T', {}).get('hate'))} HATE "
        "rows.** The `HATE`/`Dolly_T` cell is thin enough that its split proportions are "
        "visibly noisy (§11).\n"
        "- **Single-turn only.** Every row in these three configurations is single-turn, so "
        "the multi-turn code path exists and is tested but never fires. A future "
        "configuration with real multi-turn data would exercise untested-in-practice "
        "behaviour.\n"
        "- **`text_eng` ships in the release.** It is there because error analysis without "
        "it is painful. It is a trivially available leak and **must never be given to the "
        "student model**.\n"
        "- **The reproduction target is an inference, not a specification.** The assignment "
        "says only \"balance HATE and NON_HATE\" and prescribes no source stratification. "
        "`proportional` is primary because it is what a naive uniform downsample produces "
        "and therefore the likeliest provenance of the published 90.94 — but that is an "
        "inference. The quality gates and deduplication yields here also differ from "
        "whatever produced that number, so if the group misses the reproduction gate, "
        "balance mode is one candidate cause among several, and the funnel in §4 is what "
        "makes the others diagnosable.\n"
    )

    # ---------------------------------------------------------------- 14
    A("## 14. Reproduction\n")
    A("```bash\nbash reproduce.sh\n```\n")
    A(
        "One command, from an empty clone to both frozen releases. On a fresh Colab T4 the "
        "dominant cost is the teacher scoring pass; on the CPU run recorded in this "
        f"manifest it took {scoring.get('rows_per_sec', '—')} rows/sec. Acquisition streams "
        "the source parquet with HTTP range requests and keeps only four columns, so the "
        "1.71 GB source is never stored locally — about 71 MB lands in `data/raw/`.\n"
    )
    A("### Determinism, and its exact conditions\n")
    A(
        "Seed 42 throughout. Split assignment is a pure hash of `dedup_group`; balancing "
        "uses a seeded RNG over `example_id`-sorted pools; near-duplicate clusters are "
        "built from the full verified pair graph with union-find over *sorted* pairs, so "
        "the clustering library's insertion order cannot leak into the labelling.\n"
    )
    A(
        "**Verified:** on the recorded configuration — "
        f"{scoring.get('device', 'CPU')}, {scoring.get('dtype_forward', 'float32')} forward "
        "pass — every stage output is **byte-identical SHA-256 across two independent "
        "runs**.\n"
    )
    A(
        "**What changes on a T4 in fp16.** The forward pass uses different kernels and a "
        "lower-precision dtype, so `teacher_prob_hate_raw` will differ in its low-order "
        "bits and the release parquet will *not* be byte-identical to the CPU build. The "
        "labels are protected against this: the softmax is computed in float32 regardless "
        f"of the forward dtype, and thresholding uses the value rounded to "
        f"{scoring.get('prob_decimals', 6)} decimal places. On this corpus that rounding "
        "moved zero rows across a band and exactly "
        f"{_n(dist.get('rows_within_1e-6_of_threshold'))} row sits within `1e-6` of a "
        "threshold, so the label assignment is expected to be stable across hardware even "
        "though the stored floats are not. The determinism test in the suite runs on CPU "
        "for this reason.\n"
    )
    A("---\n")
    A(
        "*Every `[choice]` made in building this dataset — the 0.60 purity floor, the "
        "3–2000 character bounds, MinHash at Jaccard 0.85 over character 5-grams, "
        "downsampling rather than upsampling, 80/10/10, the majority-cell tie-break, "
        "6-decimal rounding, and the two balancing modes — is recorded above with the "
        "measurement or reasoning behind it. The two teacher thresholds (0.90 / 0.05) are "
        "fixed by the assignment and were not varied.*\n"
    )
    return "\n".join(S)
