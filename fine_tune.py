"""Bounded local Chronos-2 LoRA on economic simulation only.

Run with GARP-EFM's existing Python environment, which already contains PEFT.
The script cannot silently fall back to full fine-tuning, never loads NYC
validation/test rows, and exports standalone checkpoints for the main runtime.
Preparation is the default; actual optimization requires --run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHTS_SHA = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CONFIG_SHA = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots" / REVISION
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")
LORA_TARGETS = ["self_attention.q", "self_attention.k", "self_attention.v",
                "self_attention.o", "output_patch_embedding.output_layer"]
SEED = 20261026
STEPS = (30, 60)
CONTEXT = 256
HORIZON = 9
BATCH_SIZE = 8
EXPECTED_TRAINABLE = 1206912


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_versions():
    versions = {}
    for name in ["chronos-forecasting", "torch", "transformers", "peft", "accelerate", "numpy"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required package {name!r} is missing. Run with the existing GARP-EFM environment; full-tune fallback is forbidden.") from exc
    expected = {"chronos-forecasting": "2.3.1", "torch": "2.10.0",
                "transformers": "4.57.6", "peft": "0.18.1", "accelerate": "1.14.0"}
    for name, version in expected.items():
        if versions[name] != version:
            raise RuntimeError(f"Expected audited GARP runtime {name}=={version}, found {versions[name]}; do not mutate shared environments.")
    return versions


def load_corpus(path):
    """Strict NPZ contract; retain low-count NaNs instead of inventing prices."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"fare", "count", "timestamp_ns", *CALENDAR}
        if not required.issubset(archive.files):
            raise ValueError(f"Synthetic corpus missing {required - set(archive.files)}")
        arrays = {name: archive[name].copy() for name in required}
    fare = arrays["fare"]
    if fare.ndim != 2 or fare.shape[1] < CONTEXT + HORIZON:
        raise ValueError("Need synthetic episode-by-time matrix with >=265 hourly steps")
    for name, values in arrays.items():
        if values.shape != fare.shape:
            raise ValueError(f"Shape mismatch for {name}: {values.shape} vs {fare.shape}")
    if not np.isfinite(arrays["count"]).all() or np.any(arrays["count"] < 0):
        raise ValueError("Synthetic counts must be finite and nonnegative")
    if np.isinf(fare).any() or np.any(fare[np.isfinite(fare)] <= 0):
        raise ValueError("Synthetic fare must be positive or missing")
    if not np.isnan(fare[arrays["count"] < 10]).all():
        raise ValueError("Low-count synthetic targets must be masked, not fabricated")
    if np.any(np.isfinite(fare).sum(axis=1) < 48):
        raise ValueError("Every simulation trajectory must have at least 48 observed fare hours")
    if not np.all(np.diff(arrays["timestamp_ns"], axis=1) == 3_600_000_000_000):
        raise ValueError("Synthetic episode timestamps must have a continuous hourly grid")
    for name in CALENDAR:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Calendar covariate {name} is not finite")
    return arrays


def prepared_inputs(arrays):
    from chronos.chronos2.preprocess import from_list_of_dicts
    items = []
    for i in range(len(arrays["fare"])):
        historical = {name: arrays[name][i].astype(np.float32) for name in CALENDAR}
        historical["served_count"] = arrays["count"][i].astype(np.float32)
        items.append({"target": arrays["fare"][i].astype(np.float32)[None, :],
                      "past_covariates": historical})
    prepared = from_list_of_dicts(items, prediction_length=HORIZON,
                                 known_covariates_names=list(CALENDAR), validate_inputs=True)
    for item in prepared:
        if (item["n_targets"], item["n_covariates"], item["n_future_covariates"]) != (1, 6, 5):
            raise ValueError("Preprocessing changed the one-target, one-past-only, five-known-future contract")
    return prepared


def fixed_prediction_input(arrays):
    """First episode, fixed origin; a finite-value check, never model selection."""
    historical = {name: arrays[name][0, :CONTEXT].astype(np.float32) for name in CALENDAR}
    historical["served_count"] = arrays["count"][0, :CONTEXT].astype(np.float32)
    future = {name: arrays[name][0, CONTEXT:CONTEXT+HORIZON].astype(np.float32) for name in CALENDAR}
    return {"target": arrays["fare"][0, :CONTEXT].astype(np.float32)[None, :],
            "past_covariates": historical, "future_covariates": future}


def freeze_protocol(corpus, simulator_manifest, output, max_seconds, smoke=False):
    sim = json.loads(Path(simulator_manifest).read_text())
    arrays = load_corpus(corpus)
    versions = runtime_versions()
    if sha256(SNAPSHOT / "model.safetensors") != WEIGHTS_SHA or sha256(SNAPSHOT / "config.json") != CONFIG_SHA:
        raise ValueError("Original Chronos snapshot failed SHA-256 validation")
    protocol = {"version": "1.0", "purpose": "Simulation-only economic prior adaptation; not structural estimation",
        "smoke_only": smoke,
        "corpus": str(Path(corpus).resolve()), "corpus_sha256": sha256(corpus),
        "simulation_manifest": str(Path(simulator_manifest).resolve()),
        "simulation_manifest_sha256": sha256(simulator_manifest),
        "simulation_metadata": sim, "source_sha256": sha256(Path(__file__)),
        "model_id": "amazon/chronos-2", "base_revision": REVISION,
        "base_weights_sha256": WEIGHTS_SHA, "base_config_sha256": CONFIG_SHA,
        "device": "cpu", "dtype": "float32", "threads": 4, "runtime": versions,
        "seed": SEED, "optimizer_steps": max(STEPS), "checkpoint_steps": list(STEPS),
        "context_hours": CONTEXT, "prediction_length": HORIZON, "min_past": CONTEXT,
        "requested_batch_size_time_series": BATCH_SIZE,
        "actual_batch_time_series": 14, "tasks_per_step": 2,
        "batch_note": "Native sampler appends whole seven-channel tasks until requested batch size 8 is met, hence two tasks / fourteen series.",
        "learning_rate": 1e-5, "optimizer": "adamw_torch", "scheduler": "linear",
        "warmup_steps": 0, "weight_decay": 0., "max_gradient_norm": 1.,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0., "bias": "none", "target_modules": LORA_TARGETS},
        "expected_trainable_parameters": EXPECTED_TRAINABLE,
        "target": "Synthetic realized median rider fare, with low-count target hours masked",
        "past_only_covariates": ["served_count"], "known_future_covariates": list(CALENDAR),
        "trajectories": int(arrays["fare"].shape[0]), "hours_per_trajectory": int(arrays["fare"].shape[1]),
        "sampler": "Uniform trajectory sampling with replacement and random contiguous cut; not all passed trajectories are guaranteed observed in 60 steps",
        "training_data_policy": "Only the synthetic NPZ is loaded. No NYC validation or test panel enters gradient updates.",
        "selection_policy": ("One-step runtime smoke only; never eligible for real validation/test model selection" if smoke else
            "Root benchmark evaluates step0,30,60 on February validation only before selecting one for March. No test-driven changes."),
        "maximum_training_seconds": max_seconds, "mixed_precision": False,
        "full_finetune_forbidden": True, "resume_supported": False}
    path = output / "fine_tuning_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Frozen fine-tuning protocol differs; use a new output directory, do not overwrite evidence")
    else:
        save_json(path, protocol)
    return arrays, protocol


def run(args):
    global STEPS
    if args.smoke:
        STEPS = (1,)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "fine_tuning_manifest.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") != "prepared":
        raise RuntimeError("An optimization attempt already exists here. Resume/retraining is not implicit.")
    simulator_manifest = args.simulator_manifest or args.corpus.parent / "manifest.json"
    arrays, protocol = freeze_protocol(args.corpus, simulator_manifest, output, args.max_seconds, args.smoke)
    state = {"status": "prepared", "prepared_utc": utc_now(), "protocol_sha256": sha256(output / "fine_tuning_protocol.json"),
             "optimizer_steps_completed": 0, "checkpoint_exports": [], "training_rows_from_real_validation_or_test": 0}
    save_json(state_path, state)
    print(json.dumps({"event": "PREPARED", "trajectories": protocol["trajectories"], "steps": max(STEPS),
                      "context": CONTEXT, "device": "cpu", "run_requested": args.run}), flush=True)
    if not args.run:
        return state
    import torch
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, PeftModel
    from transformers import TrainerCallback, set_seed
    torch.set_num_threads(4)
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    inputs = prepared_inputs(arrays)
    smoke_input = fixed_prediction_input(arrays)
    pipeline = Chronos2Pipeline.from_pretrained(str(SNAPSHOT), local_files_only=True,
        device_map="cpu", dtype=torch.float32)
    pipeline.model.eval()
    # An independent copy is essential: state_dict() alone aliases live tensors
    # and could miss accidental in-place mutation of shared base storage.
    original_state = {name: value.detach().cpu().clone() for name, value in pipeline.model.state_dict().items()}
    with torch.inference_mode():
        original_q, _ = pipeline.predict_quantiles([smoke_input], prediction_length=HORIZON,
            quantile_levels=[.1, .5, .9], context_length=CONTEXT, batch_size=8, cross_learning=False)
    clock = time.monotonic()

    def verify_frozen_base(model):
        tensors = {name.replace(".base_layer.", "."): tensor for name, tensor in model.base_model.model.state_dict().items()
                   if ".lora_" not in name}
        if set(tensors) != set(original_state):
            raise ValueError("Frozen base tensor names changed")
        if any(not torch.equal(value, tensors[name].detach().cpu()) for name, value in original_state.items()):
            raise ValueError("Original base weights were modified")

    class Recorder(TrainerCallback):
        def on_train_begin(self, training_args, trainer_state, control, model=None, optimizer=None, **kwargs):
            if not isinstance(model, PeftModel) or isinstance(model.base_model.model, PeftModel):
                raise ValueError("Exactly one fresh PEFT wrapper required; full training is forbidden")
            trainables = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
            if any("lora_" not in n for n,p in trainables) or sum(p.numel() for n,p in trainables) != EXPECTED_TRAINABLE:
                raise ValueError("Unexpected trainable weights; only audited LoRA parameters may change")
            if str(training_args.device) != "cpu" or any(p.dtype != torch.float32 for n,p in trainables):
                raise ValueError("CPU/float32 training contract violated")
            if optimizer is not None and len(optimizer.state):
                raise ValueError("Expected fresh optimizer state")
            model.peft_config["default"].base_model_name_or_path = "amazon/chronos-2"
            model.peft_config["default"].revision = REVISION
            verify_frozen_base(model)
            was_training = model.training
            model.eval()
            with torch.inference_mode():
                q, _ = Chronos2Pipeline(model=model).predict_quantiles([smoke_input], prediction_length=HORIZON,
                    quantile_levels=[.1,.5,.9], context_length=CONTEXT, batch_size=8, cross_learning=False)
            if not torch.allclose(q[0], original_q[0], rtol=1e-5, atol=1e-5):
                raise ValueError("Zero-initialized LoRA changed original forecast")
            model.train(was_training)
            state.update(status="training", started_utc=utc_now(), trainable_parameters=EXPECTED_TRAINABLE,
                         zero_adapter_matches_original=True)
            save_json(state_path, state)

        def on_step_begin(self, training_args, trainer_state, control, **kwargs):
            if time.monotonic() - clock > args.max_seconds:
                raise TimeoutError("Bounded local training time exhausted")
            self.step_start = time.monotonic()

        def on_step_end(self, training_args, trainer_state, control, model=None, **kwargs):
            step = int(trainer_state.global_step)
            state["optimizer_steps_completed"] = step
            state.setdefault("step_seconds", []).append(time.monotonic() - self.step_start)
            if step in STEPS:
                verify_frozen_base(model)
                adapter = output / "adapters" / f"step{step}"
                model.save_pretrained(adapter, safe_serialization=True)
                state.setdefault("saved_adapter_steps", []).append(step)
                print(json.dumps({"event": "SAVED_ADAPTER", "step": step, "elapsed_seconds": time.monotonic()-clock}), flush=True)
            if step % 10 == 0 or step in STEPS:
                save_json(state_path, state)

        def on_log(self, training_args, trainer_state, control, logs=None, **kwargs):
            if not logs:
                return
            numeric = [v for v in logs.values() if isinstance(v, (int,float))]
            if not all(np.isfinite(v) for v in numeric):
                raise ValueError("Non-finite optimizer log")
            record = {"step": int(trainer_state.global_step), **logs}
            with (output / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({"event": "TRAIN", **record}), flush=True)

    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias="none", target_modules=LORA_TARGETS)
    try:
        tuned = pipeline.fit(inputs, prediction_length=HORIZON, context_length=CONTEXT, min_past=CONTEXT,
            finetune_mode="lora", lora_config=config, num_steps=max(STEPS), batch_size=BATCH_SIZE,
            learning_rate=1e-5, output_dir=output / "trainer", finetuned_ckpt_name="adapter_final",
            callbacks=[Recorder()], optim="adamw_torch", lr_scheduler_type="linear", warmup_steps=0,
            weight_decay=0., max_grad_norm=1., dataloader_num_workers=0, dataloader_pin_memory=False,
            bf16=False, fp16=False, tf32=False, use_cpu=True, report_to="none", save_strategy="no",
            logging_steps=10, logging_nan_inf_filter=False, disable_tqdm=True, seed=SEED, data_seed=SEED)
        verify_frozen_base(tuned.model)
        if state["optimizer_steps_completed"] != max(STEPS):
            raise ValueError("Training did not reach the prespecified final step")
        state.update(original_base_weights_unchanged=True, optimization_seconds=time.monotonic()-clock)
        del tuned, pipeline
        gc.collect()
        for step in STEPS:
            base = Chronos2Pipeline.from_pretrained(str(SNAPSHOT), local_files_only=True, device_map="cpu", dtype=torch.float32)
            adapted = PeftModel.from_pretrained(base.model, output / "adapters" / f"step{step}", is_trainable=False)
            adapted.eval()
            with torch.inference_mode():
                adapted_q, _ = Chronos2Pipeline(model=adapted).predict_quantiles([smoke_input],
                    prediction_length=HORIZON, quantile_levels=[.1,.5,.9], context_length=CONTEXT,
                    batch_size=8, cross_learning=False)
            merged = adapted.merge_and_unload(safe_merge=True)
            checkpoint = output / "checkpoints" / f"step{step}"
            merged.save_pretrained(checkpoint, safe_serialization=True)
            del merged, adapted, base
            gc.collect()
            reload = Chronos2Pipeline.from_pretrained(str(checkpoint), local_files_only=True,
                device_map="cpu", dtype=torch.float32)
            with torch.inference_mode():
                q, _ = reload.predict_quantiles([smoke_input], prediction_length=HORIZON,
                    quantile_levels=[.1,.5,.9], context_length=CONTEXT, batch_size=8, cross_learning=False)
            if not torch.isfinite(q[0]).all() or not torch.allclose(q[0], adapted_q[0], rtol=1e-4, atol=1e-4):
                raise ValueError("Merged standalone checkpoint differs from PEFT prediction")
            receipt = {"step": step, "directory": str(checkpoint),
                "weights_sha256": sha256(checkpoint / "model.safetensors"),
                "config_sha256": sha256(checkpoint / "config.json"),
                "reload_finite": True, "merged_matches_adapter": True,
                "base_weights_sha256": WEIGHTS_SHA, "fine_tuning_protocol_sha256": state["protocol_sha256"]}
            save_json(checkpoint / "provenance.json", receipt)
            state["checkpoint_exports"].append(receipt)
            save_json(state_path, state)
            del reload
            gc.collect()
        state.update(status="completed", finished_utc=utc_now(), total_seconds=time.monotonic()-clock)
        save_json(state_path, state)
        print(json.dumps({"event": "COMPLETE", "seconds": state["total_seconds"],
                          "checkpoints": [x["directory"] for x in state["checkpoint_exports"]]}), flush=True)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", stopped_utc=utc_now(), elapsed_seconds=time.monotonic()-clock)
        save_json(state_path, state)
        raise
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--simulator-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/synthetic")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the frozen 60-step local LoRA training")
    parser.add_argument("--smoke", action="store_true", help="One-step runtime smoke only; use a separate output directory; not a model-selection candidate")
    run(parser.parse_args())
