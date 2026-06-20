"""Single-GPU RoLA LM training via the canonical HF Trainer + datasets (run_clm recipe):
load corpus -> GPT-2 BPE tokenize -> group into block_size -> Trainer (loop/optim/sched/
ckpt/eval are HF's, not hand-rolled). RoLA is an AutoModel-registered HF model (rola_hf),
so it plugs straight in; the saved checkpoint runs in lm-evaluation-harness as-is.

  python train_lm.py --rola_instance rola-rla-asym --hidden_size 512 --num_hidden_layers 12 \
     --num_states 16 --d_qk 16 --d_v 16 --num_heads 8 --max_steps 20000

For SCALED multi-GPU FineWeb pretraining use flame (torchtitan) instead; this is the cheap
single-GPU path (WikiText-103 first).
"""
import argparse, math, os, json, time, tarfile, tempfile
from itertools import chain
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback
import fla_rola.models.rola  # registers RoLA with HF AutoClasses
from fla_rola.models.rola import RoLAConfig
import arms  # the 5-arm iso-backbone registry (RoLA x3 + GDN + attn ceiling)
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, default_data_collator)


FINEWEB = "HuggingFaceFW/fineweb-edu"  # natural text -> matches lm-eval's detokenized format


def _group(ex, block_size):
    # Canonical run_clm group_texts: concatenate + chunk EVERY key consistently (drops the
    # per-batch remainder — negligible at LM scale).
    cat = {k: list(chain(*ex[k])) for k in ex.keys()}
    n = (len(cat["input_ids"]) // block_size) * block_size
    res = {k: [t[i:i + block_size] for i in range(0, n, block_size)] for k, t in cat.items()}
    res["labels"] = [x[:] for x in res["input_ids"]]
    return res


def build_dataset(name, config, block_size, tokenizer, num_proc=8, max_train_samples=None):
    """Large web corpora (FineWeb-Edu) are STREAMED — never materialized to disk (the [Errno 28]
    'no space left' failure: sample-10BT is ~10B tokens). Tokenize + group into block_size on the
    fly; carve a small validation head from the stream (FineWeb-Edu ships train-only). Small datasets
    (WikiText) keep the simple materialized path."""
    if name == "listops":                                # synthetic non-reciprocal task (asym vs sym)
        from listops import listops_dataset
        n_train = int(max_train_samples) if max_train_samples else 100000
        return listops_dataset(tokenizer, n_train=n_train, n_val=2000, max_len=block_size)

    fineweb = name in ("fineweb", "fineweb-edu") or (name or "").startswith("HuggingFaceFW")
    if name in ("fineweb", "fineweb-edu"):
        name = FINEWEB
        config = config if (config or "").startswith("sample") else "sample-10BT"
    pg19 = name in ("pg19", "emozilla/pg19", "deepmind/pg19")   # long-DOCUMENT corpus (books); stream (big)
    if pg19:
        name, config = "emozilla/pg19", None

    def tok(ex):
        return tokenizer(ex["text"])

    if fineweb or pg19:                                  # stream large corpora; never materialize to disk
        raw = load_dataset(name, config, split="train", streaming=True)
        cols = list(raw.column_names) if getattr(raw, "column_names", None) else ["text"]
        blocks = (raw.map(tok, batched=True, remove_columns=cols)
                     .map(lambda ex: _group(ex, block_size), batched=True))
        N_VAL = 256                              # ~256 blocks held out for val (block_size tokens each)
        return {"train": blocks.skip(N_VAL), "validation": blocks.take(N_VAL)}

    # small/local datasets (e.g. WikiText, TinyStories): materialize (cheap), real validation split.
    config = config or None                              # TinyStories & friends have no config name
    from datasets import DatasetDict
    train_split = f"train[:{max_train_samples}]" if max_train_samples else "train"
    train_raw = load_dataset(name, config, split=train_split)
    try:
        val_raw = load_dataset(name, config, split="validation")
    except Exception:
        sp = train_raw.train_test_split(test_size=min(2000, max(2, len(train_raw) // 50)), seed=0)
        train_raw, val_raw = sp["train"], sp["test"]
    raw = DatasetDict(train=train_raw, validation=val_raw)
    cols = raw["train"].column_names
    t = raw.map(tok, batched=True, remove_columns=cols, num_proc=num_proc, desc="tokenize")
    return t.map(lambda ex: _group(ex, block_size), batched=True, num_proc=num_proc, desc=f"group {block_size}")


# ---- preemption-safe checkpoint sync (see fleet pull_in_progress) -------------------------------
# The box exposes ONLY complete checkpoints at $FLEET_CKPT_DIR/<run_id>.pt: we tar the latest HF
# checkpoint dir to a temp file then ATOMICALLY rename it into place (a reader sees either the old
# full tar or the new full tar, never a torn one). The dispatcher pulls that tar periodically and
# ships it back on re-dispatch as $FLEET_PAYLOAD_DIR/<run_id>, which we untar + resume from.
def _tar_dir_atomic(src_dir, dest_pt):
    """tar `src_dir` -> `dest_pt` via temp + atomic rename (dest is the box's served <run_id>.pt)."""
    d = os.path.dirname(dest_pt) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tar.tmp"); os.close(fd)
    with tarfile.open(tmp, "w") as t:
        t.add(src_dir, arcname=os.path.basename(src_dir))
    os.replace(tmp, dest_pt)                              # atomic on same fs


class CkptTarCallback(TrainerCallback):
    """After each HF checkpoint save, tar the LATEST checkpoint-* dir to <ckpt_dir>/<run_id>.pt so the
    box serves only a COMPLETE checkpoint (the callback fires after the save finishes)."""
    def __init__(self, out_dir, dest_pt):
        self.out_dir, self.dest_pt = out_dir, dest_pt
    def on_save(self, args, state, control, **kw):
        ck = get_last_checkpoint(self.out_dir)
        if ck:
            try:
                _tar_dir_atomic(ck, self.dest_pt)
                print(f"[ckpt-sync] tarred {os.path.basename(ck)} -> {self.dest_pt}", flush=True)
            except Exception as e:
                print(f"[ckpt-sync] WARN tar failed: {e!r}", flush=True)


def _ckpt_config_matches(ck, cfg):
    """True iff the checkpoint's config.json matches the current model on the architecture-defining
    fields. Guards against the run_id-collision gotcha: run_id = '<arm>_s<seed>' encodes neither
    geometry nor dataset, so a checkpoint from a DIFFERENT regime (e.g. d_model=512 vs 256) can land
    at the same id; resuming from it would train garbage (or crash on shape mismatch). Refuse it."""
    try:
        cj = json.load(open(os.path.join(ck, "config.json")))
    except Exception:
        return False
    for k in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size", "num_states", "d_qk", "d_v"):
        want = getattr(cfg, k, None)
        if want is not None and k in cj and cj[k] != want:
            print(f"[resume] checkpoint {os.path.basename(ck)} config mismatch: {k}={cj[k]} != {want} "
                  f"-> REFUSING this checkpoint, starting FRESH (run_id collision across regimes?)", flush=True)
            return False
    return True


def _restore_resume(run_id, out_dir, cfg):
    """Resolve a checkpoint for Trainer.resume_from_checkpoint, in priority order:
      1. SHIPPED TAR ($FLEET_PAYLOAD_DIR/<run_id>): the dispatcher pulled this off a dead box and
         shipped it back on re-dispatch -> untar into out_dir + resume. FAIL LOUD if corrupt.
      2. LOCAL DISK (out_dir already holds a checkpoint-*): the box_server restarted on a PERSISTENT
         disk (on-demand pause+resume, or any container restart) -> resume from local, no transfer.
    Either candidate is REFUSED (fresh start) if its config doesn't match `cfg` — see
    _ckpt_config_matches. Returns the checkpoint dir or None (fresh start)."""
    pdir = os.environ.get("FLEET_PAYLOAD_DIR")
    tar = os.path.join(pdir, run_id) if pdir else None
    if tar and os.path.exists(tar) and os.path.getsize(tar) > 0:
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar, "r") as t:
            t.extractall(out_dir)
        ck = get_last_checkpoint(out_dir)
        if ck is None or not os.path.exists(os.path.join(ck, "trainer_state.json")):
            raise RuntimeError(f"resume tar {tar} present but no valid checkpoint after untar (corrupt) "
                               f"— refusing to silently restart; re-dispatch will retry")
        if not _ckpt_config_matches(ck, cfg):            # stale/wrong ship (run_id collision) -> fresh
            return None
        print(f"[resume] restored {os.path.basename(ck)} from shipped tar -> resuming", flush=True)
        return ck
    # No tar shipped: resume from a checkpoint already on the LOCAL disk (paused+resumed / restarted
    # box). This is the transfer-free recovery path the no-kill dispatcher relies on.
    if os.path.isdir(out_dir):
        ck = get_last_checkpoint(out_dir)
        if ck is not None and os.path.exists(os.path.join(ck, "trainer_state.json")) and _ckpt_config_matches(ck, cfg):
            print(f"[resume] found local checkpoint {os.path.basename(ck)} in {out_dir} -> resuming "
                  f"(no transfer)", flush=True)
            return ck
    return None


def main():
    p = argparse.ArgumentParser()
    # --arm selects from the 5-arm iso-backbone registry (arms.ARMS): the canonical RoLA-vs-attention
    # LM comparison. When set, it OVERRIDES the legacy --rola_instance/geometry flags below (those
    # remain for ad-hoc single-RoLA runs). Fleet contract: --arm + --results + --ckpt_dir.
    p.add_argument("--arm", default=None, choices=(None, *arms.ARMS),
                   help="registry arm (rola-rla-kappa-{sym,asym} / rola-gla-kappa-asym / gdn / attn)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--results", default=None, help="fleet result ndjson to append {run_id, ok, ...}")
    p.add_argument("--ckpt_dir", default=os.environ.get("FLEET_CKPT_DIR"), help="dir to save_pretrained into")
    p.add_argument("--rola_instance", default="rola-rla-asym")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_states", type=int, default=16)
    p.add_argument("--d_qk", type=int, default=16)
    p.add_argument("--d_v", type=int, default=16)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000, help="periodic checkpoint cadence (preemption-resume)")
    p.add_argument("--grad_ckpt", type=int, default=0, help="gradient checkpointing (1=on, frees activation mem for bigger batch)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="slice train to N docs (calibration / token-budgeted runs)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    # run_id: arm path -> "<arm>_s<seed>" (fleet work-item id); legacy path -> the old descriptive id.
    run_id = f"{a.arm}_s{a.seed}" if a.arm else f"{a.rola_instance}-d{a.hidden_size}L{a.num_hidden_layers}"
    out = a.out_dir or (os.path.join(a.ckpt_dir, run_id) if a.ckpt_dir else f"lm_runs/{run_id}")

    # text8 is CHAR-level (27-symbol vocab, no GPT-2 BPE): pre-grouped LM blocks, default collator,
    # bits-per-char metric. Every other corpus uses the GPT-2 tokenizer + build_dataset.
    is_char = (a.dataset == "text8")
    if is_char:
        import text8_data
        tok = None; vocab_sz = text8_data.vocab_size(); eos_id = 0
        ds = text8_data.text8_dataset(a.block_size, max_train_blocks=a.max_train_samples)
    else:
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        vocab_sz = len(tok); eos_id = tok.eos_token_id
        ds = build_dataset(a.dataset, a.dataset_config, a.block_size, tok, a.num_proc, a.max_train_samples)
    if a.arm:                                            # iso-backbone registry arm (the LM comparison)
        model = arms.build_arm(a.arm, vocab_size=vocab_sz, bos_token_id=eos_id, eos_token_id=eos_id)
        state_floats = arms.recurrent_state_floats(a.arm)
    else:                                                # legacy single-RoLA run
        cfg = RoLAConfig(rola_instance=a.rola_instance, num_states=a.num_states, d_qk=a.d_qk,
                         d_v=a.d_v, num_heads=a.num_heads, hidden_size=a.hidden_size,
                         num_hidden_layers=a.num_hidden_layers, vocab_size=vocab_sz,
                         bos_token_id=eos_id, eos_token_id=eos_id)
        model = AutoModelForCausalLM.from_config(cfg)
        state_floats = a.num_heads * a.num_states * a.d_qk * (a.d_v + 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {run_id} {n_params/1e6:.1f}M params  state/layer={state_floats}", flush=True)

    # transformers 5.x expects each module's _tied_weights_keys to be a dict {tied: source}; fla's
    # CausalLM classes (GatedDeltaNet, Transformer) still declare it as a LIST (e.g. ['lm_head.weight']),
    # which crashes save_pretrained -> _get_tied_weight_keys ('.keys()' on a list) at the first ckpt.
    # We use tie_word_embeddings=False (untied) so nothing is actually tied: normalize any list-valued
    # _tied_weights_keys to {} across the module tree. RoLA already uses {}, so this is a no-op there.
    for _mod in model.modules():
        if isinstance(getattr(_mod, "_tied_weights_keys", None), list):
            _mod._tied_weights_keys = {}

    # Resume tar shipped by the dispatcher (preemption-safe). Resolve BEFORE building the Trainer so
    # output_dir holds the restored checkpoint for resume_from_checkpoint. ckpt_pt is the box-served
    # <run_id>.pt the dispatcher pulls in-progress.
    resume_ckpt = _restore_resume(run_id, out, model.config)
    ckpt_pt = os.path.join(a.ckpt_dir, run_id + ".pt") if a.ckpt_dir else os.path.join(out + ".pt")

    args = TrainingArguments(
        output_dir=out, max_steps=a.max_steps, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=min(a.warmup, max(10, a.max_steps // 50)),   # cap warmup at ~2% (reduced runs)
        adam_beta1=0.9, adam_beta2=0.95,        # modern-LM standard (GPT-3/Llama/Mamba/GLA); HF default beta2=0.999
        seed=a.seed, weight_decay=a.weight_decay, max_grad_norm=1.0, bf16=True,
        remove_unused_columns=False,            # streaming IterableDataset: keep our token columns intact
        # grad-ckpt: env-tunable. OFF by default (peak 6.9GB/40GB at batch 8 — recompute was pure
        # slowdown). ON frees the routed-RoLA activation memory so a bigger batch fits (better GPU
        # utilization) — A/B via LM_GRAD_CKPT. (RoLA appears kernel-bound, so batch may not help it.)
        gradient_checkpointing=bool(a.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.grad_ckpt else None,
        logging_steps=50, eval_strategy="steps", eval_steps=a.eval_steps,
        # PREEMPTION-SAFE: periodic checkpoints every save_steps, keep last 2 (defense vs a torn
        # newest). save_safetensors=False -> .bin so the save doesn't crash on sym's tied/shared
        # routers (safetensors refuses shared tensors — the reason this used to be save_strategy=no).
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
        # (transformers 5.x removed the `save_safetensors` arg — safetensors is the default save.
        # RoLA-sym ties the router via read_router=None, so there's a single router weight, NOT two
        # state_dict keys sharing one tensor; embeddings are untied -> no shared-tensor save crash.)
        report_to="none", dataloader_num_workers=4)
    # ListOps pre-builds answer-only labels (-100 except the answer token); the LM collator would
    # OVERWRITE them with full-sequence next-token labels -> use a passthrough collator so the eval
    # loss is the ANSWER NLL it's meant to be, and report answer-token exact-match accuracy (the
    # headline metric — NLL alone is gameable by the per-op answer prior).
    is_listops = (a.dataset == "listops")
    pre_grouped = is_listops or is_char          # both ship fixed-length input_ids+labels -> passthrough collator
    def _answer_acc(eval_pred):
        preds, labels = eval_pred                     # preds = argmax ids [N,L] (preprocess fn below)
        labels, preds = labels[:, 1:], preds[:, :-1]  # causal shift: preds[t] predicts labels[t+1]
        m = labels != -100
        return {"answer_acc": float((preds[m] == labels[m]).mean()) if m.any() else 0.0}
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"],
                      data_collator=(default_data_collator if pre_grouped
                                     else DataCollatorForLanguageModeling(tok, mlm=False)),
                      compute_metrics=_answer_acc if is_listops else None,
                      preprocess_logits_for_metrics=((lambda logits, lab: logits.argmax(-1))
                                                     if is_listops else None),
                      callbacks=[CkptTarCallback(out, ckpt_pt)])   # tar each checkpoint -> box-served <run_id>.pt
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tr = trainer.train(resume_from_checkpoint=resume_ckpt)
    metrics = trainer.evaluate()
    ppl = math.exp(min(metrics["eval_loss"], 20))
    bpc = round(metrics["eval_loss"] / math.log(2), 4) if is_char else None   # text8: bits-per-char
    # Resource profile (the cost axes RoLA actually spends on — reported alongside quality):
    tok_per_s = round(tr.metrics.get("train_samples_per_second", 0.0) * a.block_size, 1)
    peak_mib = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
    answer_acc = metrics.get("eval_answer_acc")           # listops only (None otherwise)
    print(f"DONE {run_id} eval_loss={metrics['eval_loss']:.4f} ppl={ppl:.2f} "
          f"{'bpc=%.4f ' % bpc if bpc is not None else ''}"
          f"{'answer_acc=%.4f ' % answer_acc if answer_acc is not None else ''}"
          f"tok/s={tok_per_s} peak_mib={peak_mib}", flush=True)
    # Final EVAL artifact: a clean save_pretrained dir (model+config+tokenizer, no optimizer/checkpoints)
    # → tar to the box-served <run_id>.pt, overwriting the last resume tar. from_pretrained-able for the
    # post-hoc recall eval. .bin (safe_serialization=False) allows sym's shared routers.
    final_dir = os.path.join(out, "final")
    model.save_pretrained(final_dir, safe_serialization=False)
    if tok is not None: tok.save_pretrained(final_dir)   # char path has no HF tokenizer to save
    if a.ckpt_dir:
        try:
            _tar_dir_atomic(final_dir, ckpt_pt)
            print(f"[ckpt-sync] final model tarred -> {ckpt_pt}", flush=True)
        except Exception as e:
            print(f"[ckpt-sync] WARN final tar failed: {e!r}", flush=True)

    if a.results:   # fleet result row — recall-downstream eval is a separate post-hoc pass over the ckpt
        row = dict(run_id=run_id, ok=True, arm=a.arm or a.rola_instance, seed=a.seed,
                   params_m=round(n_params / 1e6, 2), state_floats=state_floats,
                   hidden_size=getattr(model.config, "hidden_size", None),
                   num_hidden_layers=getattr(model.config, "num_hidden_layers", None),
                   tokens=a.max_steps * a.batch_size * a.grad_accum * a.block_size,
                   eval_loss=round(metrics["eval_loss"], 4), ppl=round(ppl, 3), bpc=bpc,
                   dataset=a.dataset, block_size=a.block_size,
                   answer_acc=(round(answer_acc, 4) if answer_acc is not None else None),
                   train_tok_per_s=tok_per_s, peak_mib=peak_mib, ckpt=ckpt_pt)
        with open(a.results, "a") as fh:
            fh.write(json.dumps(row) + "\n"); fh.flush()
        print("RESULT_ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
