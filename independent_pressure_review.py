"""Saved-artifact and causal-input review of the separate pressure demonstration.

Never fits a model, changes a controller, or selects a favorable heldout subset.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pressure_benchmark as benchmark
from decision_layer import choose_action, solve_response


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT/"outputs/decision_demo"
    metrics = json.loads((out/"metrics.json").read_text())
    provenance = json.loads((out/"provenance.json").read_text())
    pred = pd.read_parquet(out/"predictions.parquet")
    surfaces = pd.read_parquet(out/"surfaces.parquet")
    decisions = pd.read_parquet(out/"decisions.parquet")
    heldout = np.load(ROOT/"outputs/synthetic/validation.npz",allow_pickle=False)
    checks={}

    def check(name, condition):
        checks[name]=bool(condition)
        if not condition:
            raise AssertionError(name)

    check("completed",provenance["status"]=="completed")
    check("original_public_weights_only",provenance["model_weight_sha256"]==json.loads((ROOT/"protocol.json").read_text())["chronos_weights_sha256"])
    check("heldout_corpus_hash",sha(ROOT/"outputs/synthetic/validation.npz")==provenance["heldout_corpus_sha256"])
    check("training_corpus_hash",sha(ROOT/"outputs/synthetic/corpus.npz")==provenance["training_corpus_sha256"])
    check("decision_source_hash",sha(ROOT/"decision_layer.py")==provenance["decision_layer_sha256"])
    check("frozen_first_16_series",provenance["heldout_series_indices"]==list(range(16)))
    check("frozen_four_origins",provenance["target_bin_indices"]==[512,640,768,896])
    check("no_duplicate_case_models",not pred.duplicated(["case_id","model"]).any())
    check("64_cases_each_model",pred.groupby("model").size().eq(64).all())
    check("same_64_case_support",pred.groupby("case_id").model.nunique().eq(pred.model.nunique()).all())
    check("eleven_action_candidates_each_case",surfaces.groupby(["case_id","model"]).size().eq(11).all())
    check("pressure_finite",np.isfinite(pred[["actual_potential","predicted_potential","actual_capacity","predicted_capacity","queue"]]).all().all())
    for row in pred.itertuples():
        s,t=int(row.series_index),int(row.target_index)
        check(f"{row.case_id}/{row.model}/previous_queue",row.queue==float(heldout["queue_end"][s,t-1]))
        check(f"{row.case_id}/{row.model}/actual_potential",row.actual_potential==float(heldout["potential"][s,t]))
        check(f"{row.case_id}/{row.model}/actual_capacity",row.actual_capacity==float(heldout["capacity"][s,t]))
    # Mutating target/current/future outcomes must not change forecast inputs.
    mutated={key:heldout[key].copy() for key in heldout.files}
    for key in ("potential","capacity"):
        mutated[key][:,512:]+=10000
    check("classical_features_ignore_target_and_future",np.array_equal(benchmark.features(heldout,0,512),benchmark.features(mutated,0,512)))
    original=benchmark.chronos_inputs(heldout,[(0,512)])[0]
    changed=benchmark.chronos_inputs(mutated,[(0,512)])[0]
    check("chronos_context_ignores_target_and_future",np.array_equal(original["target"],changed["target"]))
    check("chronos_has_exact_256h_history",original["target"].shape==(2,256))
    check("chronos_future_covariates_only_calendar",set(original["future_covariates"])==set(benchmark.CALENDAR))

    config=provenance["decision_model"]
    evaluated=[]
    for row in pred.itertuples():
        key=(decisions.case_id.eq(row.case_id)&decisions.model.eq(row.model))
        d=decisions.loc[key].iloc[0]
        forecast=choose_action(row.predicted_potential,row.predicted_capacity,row.queue,provenance["price_grid"],config=config)
        oracle=choose_action(row.actual_potential,row.actual_capacity,row.queue,provenance["price_grid"],config=config)
        action=forecast["selected_action"]
        check(f"{row.case_id}/{row.model}/forecast_feasibility",bool(d.forecast_feasible)==bool(forecast["feasible"]))
        check(f"{row.case_id}/{row.model}/oracle_feasibility",bool(d.oracle_feasible)==bool(oracle["feasible"]))
        realized=None
        if action is not None:
            check(f"{row.case_id}/{row.model}/selected_without_realized_state",d.selected_price_multiplier==action["price_multiplier"])
            realized=solve_response(row.actual_potential,row.actual_capacity,row.queue,d.selected_price_multiplier,config)
        real_feasible=bool(realized is not None and not realized["zero_capacity"] and realized["quoted_eta_proxy_minutes"]<=provenance["eta_constraint_minutes"]+1e-9)
        check(f"{row.case_id}/{row.model}/realized_input_feasibility",bool(d.realized_input_feasible)==real_feasible)
        raw_regret=None
        if oracle["selected_action"] is not None and real_feasible:
            raw_regret=oracle["selected_action"]["new_rider_completions"]-realized["new_rider_completions"]
            check(f"{row.case_id}/{row.model}/no_negative_oracle_regret_hidden",raw_regret>=-1e-8)
            check(f"{row.case_id}/{row.model}/regret_recomputed",np.isclose(d.feasible_grid_regret_new_completions,max(0,raw_regret),rtol=0,atol=1e-9))
        evaluated.append({"case_id":row.case_id,"model":row.model,"realized_feasible":real_feasible,"regret":raw_regret})
    evaluated=pd.DataFrame(evaluated)
    common=evaluated.groupby("case_id").realized_feasible.all()
    common_ids=common[common].index.tolist()
    check("identical_feasible_regret_support",common_ids==metrics["common_feasible_case_ids"])
    recomputed=[]
    for row in metrics["metrics"]:
        model=row["model"]
        p=pred[pred.model.eq(model)]
        s=surfaces[surfaces.model.eq(model)]
        d=decisions[decisions.model.eq(model)]
        a=(p.predicted_potential-p.actual_potential).abs().mean()
        c=(p.predicted_capacity-p.actual_capacity).abs().mean()
        values={"model":model,"potential_mae":a,"capacity_mae":c,"joint_state_mae":(a+c)/2,
                "eta_model_response_mae_minutes":(s.predicted_quoted_eta_proxy_minutes-s.realized_input_quoted_eta_proxy_minutes).abs().mean(),
                "request_rate_model_response_mae_pp":100*(s.predicted_request_rate-s.realized_input_request_rate).abs().mean(),
                "new_completion_conversion_model_response_mae_pp":100*(s.predicted_new_rider_conversion-s.realized_input_new_rider_conversion).abs().mean(),
                "selected_action_count":int(d.forecast_feasible.sum()),
                "abstention_count":int((~d.forecast_feasible).sum()),
                "selected_action_feasible_under_realized_inputs_count":int(d.realized_input_feasible.sum()),
                "infeasible_selected_action_count":int((d.forecast_feasible&~d.realized_input_feasible).sum()),
                "common_feasible_cases":len(common_ids)}
        for name,value in values.items():
            if name in ("model","abstention_count"):
                continue
            check(f"metric/{model}/{name}",np.isclose(value,row[name],rtol=0,atol=1e-10))
        r=evaluated.loc[evaluated.model.eq(model)&evaluated.case_id.isin(common_ids),"regret"]
        check(f"metric/{model}/regret",row["common_feasible_grid_regret_new_completions"] is None if len(r)==0 else np.isclose(r.clip(lower=0).mean(),row["common_feasible_grid_regret_new_completions"],rtol=0,atol=1e-9))
        values["common_feasible_grid_regret_new_completions"]=float(r.clip(lower=0).mean()) if len(r) else None
        recomputed.append(values)
    receipt={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"check_count":len(checks),
             "audit_script_sha256":sha(__file__),"inputs_sha256":{name:sha(out/name) for name in ("metrics.json","provenance.json","predictions.parquet","surfaces.parquet","decisions.parquet")},
             "metrics_recomputed":recomputed,
             "boundaries":["First 16 heldout trajectories, not necessarily 16 independent markets; two routes were generated per episode.",
                           "One-step plug-in decisions on fixed observed queues; not closed-loop policy evaluation over time.",
                           "Known illustrative response parameters, not estimated real-world behavioral coefficients.",
                           "Response errors compare forecast-input and realized-input calculations, not observed conversion/ETA labels.",
                           "Full-information reference is an oracle within the same limited action grid and response model.",
                           "Common-feasible regret omits failures and abstentions, which must remain separately visible.",
                           "Zero-shot Chronos is separate from the synthetic fare adapter; no demonstrated fine-tuned transfer to pressure."]}
    (out/"independent_review.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"passed","checks":len(checks),"metrics":recomputed},indent=2))


if __name__=="__main__":
    main()
