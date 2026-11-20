"""Stage admission, worker isolation, verification and selection for Slice 6."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from uuid import uuid4

from .accounting import Attempt, budget_summary, identity, merge_accounting, utc_now
from .classical_run import _campaign_lock
from .constants import CUTOFF, MODEL_ID, MODEL_REVISION
from .profiles import profile_errors
from .provenance import json_bytes, save_json, sha256_bytes, sha256_file
from .runtime import versions
from .transformer import (
    KEYS, REPEAT_KEYS, SCREEN_KEYS, do_training, resolved_run,
    select_screen, steps_per_epoch, summarize_selected, verify_epoch, verify_rolling,
)
from .transformer_budget import (
    allocation, choose_group, evaluation_reserve, overlap_rate,
    pilot_estimate, run_seconds, time_remaining,
)
from .transformer_data import read_json
from .transformer_handoff import (
    RETURN_NAME, check_index, directory_for, export_campaign, import_campaign,
    prepare_campaign, receive_campaign,
)


def _digest(value):
    return sha256_bytes(json_bytes(value))


class Campaign:
    def __init__(self, root, campaign_id, *, receipt_sha256=None):
        self.id = campaign_id
        self.directory = directory_for(root, campaign_id)
        self.state = self.directory / "state"
        self.artifacts = Path(root).resolve() / "artifacts"
        self.receipt, self.receipt_sha256, self.inputs = check_index(
            self.directory, campaign_id, receipt_sha256
        )

    def attempt_record(self, accounting, config, attempt_id):
        identity(attempt_id)
        attempt = read_json(accounting / "runs" / config["run_id"] / "attempts" / attempt_id / "attempt.json")
        meta = attempt["metadata"]
        if (attempt["run_id"] != config["run_id"] or attempt["attempt_id"] != attempt_id
                or attempt["status"] not in ("completed", "failed", "interrupted")
                or "ended_at" not in attempt or meta["config_sha256"] != _digest(config)
                or meta["code_sha256"] != self.receipt["source"]["code_sha256"]
                or meta["data_sha256"] != self.inputs.lineage["eligible_manifest_sha256"]):
            raise ValueError("Saved transformer state lacks a matching terminal attempt.")
        return attempt

    def inspect(self, *, state=None, accounting=None, checkpoints=False):
        state = self.state if state is None else state
        accounting = self.artifacts if accounting is None else accounting
        selection = None
        statuses, reports, configs = {}, {}, {}
        for key in KEYS:
            directory = state / key
            if not directory.exists():
                statuses[key] = {"status": "not_started", "steps": 0, "epochs": []}
                continue
            config = read_json(directory / "config.json")
            wanted = resolved_run(
                key, campaign=self.id, inputs=self.inputs, source=self.receipt["source"],
                runtime=config["runtime"], physical_batch=config["physical_batch"],
                device=config["device"], selection=selection,
            )
            if _digest(config) != _digest(wanted):
                raise ValueError("Run configuration differs from the approved campaign recipe.")
            if key in REPEAT_KEYS:
                selected = configs[selection["key"]]
                if any(config[field] != selected[field] for field in ("runtime", "physical_batch", "device")):
                    raise ValueError("Repeat must use the same runtime and batch settings as selected seed 42.")
            configs[key] = config
            latest = verify_rolling(directory)
            completed = [epoch for epoch in range(1, 4) if (directory / "epochs" / f"e{epoch}").is_dir()]
            expected = list(range(1, config["max_epochs"] + 1)) if key in SCREEN_KEYS else [config["max_epochs"]]
            if completed != expected[:len(completed)]:
                raise ValueError("Unexpected or out-of-order transformer epoch artifacts.")
            steps = 0
            if latest:
                steps = latest["progress"]["optimizer_steps"]
                attempt = self.attempt_record(accounting, config, latest["attempt_id"])
                if (attempt["artifact_sha256"].get("rolling") != latest["sha256"]
                        or attempt["progress"] != latest["progress"]
                        or not 0 < steps <= config["steps_per_epoch"] * config["max_epochs"]):
                    raise ValueError("Rolling state and attempt progress disagree.")
            for epoch in completed:
                selected_epoch = selection and key in (selection["key"], *REPEAT_KEYS) and epoch == selection["epoch"]
                report, receipt = verify_epoch(directory, config, self.inputs, epoch,
                                               load_checkpoint=checkpoints and bool(selected_epoch))
                attempt = self.attempt_record(accounting, config, receipt["attempt_id"])
                self.attempt_record(accounting, config, receipt["checkpoint_attempt_id"])
                if (attempt["artifact_sha256"].get(f"epoch{epoch}") !=
                        sha256_file(directory / "epochs" / f"e{epoch}" / "epoch.json")):
                    raise ValueError("Epoch publication did not match its recorded attempt.")
                reports[key, epoch] = report
            if completed and steps < completed[-1] * config["steps_per_epoch"]:
                raise ValueError("Epoch evidence cannot exceed recorded optimizer progress.")
            complete = directory / "complete.json"
            if complete.exists():
                record = read_json(complete)
                attempt = self.attempt_record(accounting, config, record["attempt_id"])
                if (completed != expected or record["last_epoch"] != config["max_epochs"]
                        or attempt["artifact_sha256"].get("complete") != sha256_file(complete)):
                    raise ValueError("Completed run is missing verified epoch evidence.")
            pilot = directory / "pilot.json"
            if pilot.exists():
                data = read_json(pilot)
                attempt = self.attempt_record(accounting, config, data["attempt_id"])
                if attempt["artifact_sha256"].get("pilot") != sha256_file(pilot):
                    raise ValueError("Pilot timing does not match its recorded attempt.")
            statuses[key] = {"status": "complete" if complete.exists() else
                             "pilot" if pilot.exists() and not completed else "incomplete",
                             "steps": steps, "epochs": completed,
                             "truncated": read_json(complete if complete.exists() else pilot)["truncated"]
                             if complete.exists() or pilot.exists() else None}
            if key == SCREEN_KEYS[-1] and all(statuses[screen]["status"] == "complete" for screen in SCREEN_KEYS):
                candidates = {(candidate, epoch): report["pooled"]["macro_f1"]
                              for (candidate, epoch), report in reports.items()}
                winner, epoch = select_screen(candidates)
                left, right = (configs[screen] for screen in SCREEN_KEYS)
                if any(left[field] != right[field] for field in ("runtime", "physical_batch", "device")):
                    raise ValueError("Both screenings must use the same runtime and batch recipe.")
                report = reports[winner, epoch]
                chosen = state / winner / "epochs" / f"e{epoch}"
                selection = {
                    "key": winner, "epoch": epoch, "learning_rate": configs[winner]["learning_rate"],
                    "scheduler_total_steps": configs[winner]["scheduler_total_steps"],
                    "pooled_macro_f1": report["pooled"]["macro_f1"],
                    "checkpoint_sha256": read_json(chosen / "epoch.json")["files"]["checkpoint.pt"],
                    "eval_config_sha256": report["bundle"]["config_sha256"],
                }
                if checkpoints:
                    verify_epoch(state / winner, configs[winner], self.inputs, epoch, load_checkpoint=True)
        cohort = None
        if selection is not None:
            chosen = [selection["key"], *(key for key in REPEAT_KEYS if statuses[key]["status"] == "complete")]
            seed_reports = {configs[key]["seed"]: reports[key, selection["epoch"]] for key in chosen}
            cohort = summarize_selected(
                seed_reports, selection,
                {seed: report["bundle"]["config_sha256"] for seed, report in seed_reports.items()},
            )
        return {
            "schema_version": 1, "campaign": self.id, "role": "model_selection",
            "freeze_id": self.inputs.lineage["freeze_id"],
            "state": statuses, "selection": selection, "cohort": cohort,
            "reports": [{"key": key, "epoch": epoch, "aggregate": report}
                        for (key, epoch), report in reports.items()],
            "full_schedule_complete": all(item["status"] == "complete" for item in statuses.values()),
        }

    def start_inputs(self):
        # Import exact pre-existing attempt bytes once, including all 24 prior attempts.
        merge_accounting(self.directory / "inputs" / "prior-accounting", self.artifacts)

    def stage_config(self, key, runtime, physical_batch, device, selection):
        config = resolved_run(
            key, campaign=self.id, inputs=self.inputs, source=self.receipt["source"],
            runtime=runtime, physical_batch=physical_batch, device=device, selection=selection,
        )
        directory = self.state / key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        if path.exists() and path.read_bytes() != json_bytes(config):
            raise ValueError("Runtime or batch changed. Prepare a new campaign for an accounted restart.")
        if not path.exists():
            save_json(path, config)
        return directory, config


def runtime_inventory(backend, *, user_privacy):
    import platform
    import torch

    values = versions()
    profile = "kaggle-py313" if backend == "cuda" else "macos-py312"
    if profile_errors(profile, values["python"], values):
        raise ValueError("Installed versions differ from the adopted profile. Run doctor and review the drift.")
    slots = (None,)
    runtime = {"profile": profile, "versions": values, "platform": platform.system(),
               "architecture": platform.machine(), "backend": backend}
    if backend == "cuda":
        if not user_privacy or not torch.cuda.is_available():
            raise ValueError("Private accelerator readiness must be confirmed before research launch.")
        devices = [torch.cuda.get_device_properties(n) for n in range(torch.cuda.device_count())]
        first = devices[0]
        signature = (first.name, first.total_memory, first.major, first.minor)
        slots = tuple(n for n, candidate in enumerate(devices)
                      if (candidate.name, candidate.total_memory, candidate.major, candidate.minor) == signature)[:2]
        runtime.update({"accelerator": first.name, "accelerator_bytes": first.total_memory,
                        "compute_capability": [first.major, first.minor], "cuda_build": torch.version.cuda})
    elif backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in the local pinned runtime.")
    return runtime, slots


def cached_model(root):
    from huggingface_hub import snapshot_download

    cache = Path(root).resolve() / ".cache" / "huggingface"
    if cache.resolve() != cache:
        raise ValueError("Pretrained model cache cannot use symbolic links.")
    return snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache,
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
    )


def _worker(slot, hard_deadline, soft_deadline, ready, go, halfway, reply, args, kwargs):
    if slot is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else None
        os.environ["CUDA_VISIBLE_DEVICES"] = str(slot) if devices is None else devices[slot]
    remaining = hard_deadline - time.monotonic()
    if remaining <= 0:
        return
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            do_training(*args, deadline=soft_deadline, ready_event=ready, go_event=go,
                        halfway_event=halfway, **kwargs)
        reply.send("ok")
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, EOFError, ImportError) as error:
        reply.send(type(error).__name__)
    finally:
        reply.close()


def _attempt_progress(attempt, directory):
    progress = None
    files = {}
    path = directory / "latest.json"
    if path.exists():
        latest = read_json(path)
        if latest["attempt_id"] == attempt.attempt_id:
            progress = latest["progress"]
            files["rolling"] = latest["sha256"]
    for name in ("pilot", "complete"):
        path = directory / f"{name}.json"
        if path.exists() and read_json(path)["attempt_id"] == attempt.attempt_id:
            files[name] = sha256_file(path)
    for epoch in range(1, 4):
        path = directory / "epochs" / f"e{epoch}" / "epoch.json"
        if path.exists() and read_json(path)["attempt_id"] == attempt.attempt_id:
            files[f"epoch{epoch}"] = sha256_file(path)
    attempt.update(progress=progress, artifact_sha256=files)


def _supervise(campaign, jobs, *, stage, model_root, seconds, reserve_seconds, overlap, quota_hours, trial):
    context = multiprocessing.get_context("spawn")
    halfway = context.Event() if overlap else None
    high_ready = context.Event() if overlap else None
    children, outcomes = {}, {}
    hard_deadline = time.monotonic() + seconds

    def launch(key, directory, config, slot, first):
        latest = verify_rolling(directory)
        attempt = Attempt(campaign.artifacts, config["run_id"], metadata={
            "config_sha256": _digest(config), "code_sha256": config["source"]["code_sha256"],
            "code_revision": config["source"]["code_revision"],
            "data_sha256": config["lineage"]["eligible_manifest_sha256"],
            "model_revision": MODEL_REVISION, "versions": config["runtime"]["versions"],
            "seed": config["seed"], "device": config["device"],
            "parent_checkpoint_sha256": latest["sha256"] if latest else None,
            "settings": {"campaign": campaign.id, "stage": stage, "physical_slot": slot,
                         "physical_batch": config["physical_batch"], "trial": trial},
            "provider_quota_before": quota_hours,
        })
        ready = high_ready if overlap and not first else None
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(
            slot, hard_deadline, hard_deadline - reserve_seconds, ready,
            high_ready if overlap and first else None,
            halfway if overlap and first else None, send,
            (directory, config, campaign.inputs, model_root, stage, attempt.attempt_id),
            {"latest": latest, "trial": trial},
        ))
        attempt.__enter__()
        try:
            process.start()
        except BaseException as error:
            attempt.__exit__(type(error), error, None)
            receive.close()
            raise
        finally:
            send.close()
        children[key] = process, receive, attempt, directory

    def finish(key, *, kill):
        process, pipe, attempt, directory = children.pop(key)
        if kill and process.is_alive():
            process.kill()
        process.join()
        status = "interrupted" if kill or process.exitcode == -signal.SIGALRM else "failed"
        if process.exitcode == 0 and pipe.poll():
            try:
                outcome = pipe.recv()
                status = "completed" if outcome == "ok" else outcome
            except EOFError:
                pass
        pipe.close()
        process.close()
        try:
            _attempt_progress(attempt, directory)
        finally:
            kind = None if status == "completed" else KeyboardInterrupt if status == "interrupted" else RuntimeError
            attempt.__exit__(kind, None, None)
        outcomes[key] = status

    try:
        launch(*jobs[0], first=True)
        if len(jobs) > 1 and not overlap:
            launch(*jobs[1], first=False)
        unlaunched = overlap
        heartbeat = 0
        while children:
            if unlaunched and halfway.is_set():
                launch(*jobs[1], first=False)
                unlaunched = False
            if time.monotonic() >= hard_deadline or utc_now() >= datetime.fromisoformat(CUTOFF):
                break
            for key, (process, _, attempt, directory) in list(children.items()):
                if not process.is_alive():
                    finish(key, kill=False)
                    if overlap and key == jobs[1][0]:
                        high_ready.set()
                elif time.monotonic() >= heartbeat:
                    _attempt_progress(attempt, directory)
            if time.monotonic() >= heartbeat:
                heartbeat = time.monotonic() + 5
            time.sleep(.20)
    finally:
        for key in tuple(children):
            finish(key, kill=True)
    return outcomes


def stage_plan(campaign, stage, snapshot, available, slots, runtime, physical_batch):
    status = snapshot["state"]
    pilots = {
        key: read_json(campaign.state / key / "pilot.json")
        for key in SCREEN_KEYS if (campaign.state / key / "pilot.json").is_file()
    }
    if stage == "pilot":
        keys = [key for key in SCREEN_KEYS if key not in pilots and status[key]["status"] != "complete"]
        target = keys[:len(slots)] if keys and keys[0] == SCREEN_KEYS[0] else keys[:1]
        return target, {"mode": stage, "remaining": available["active"], "reserve": 0,
                        "overlap": len(target) == 2}, 5
    if not pilots:
        raise ValueError("A completed timed research pilot is required before screening.")
    if stage == "repeat" and not snapshot["selection"]:
        raise ValueError("Both screenings must finish before starting the repeats.")
    sample = {key: pilot_estimate(pilot) for key, pilot in pilots.items()}
    # Use the slowest observed component, so a new independent run is forecast conservatively.
    estimate = {name: max(item[name] for item in sample.values()) for name in next(iter(sample.values()))}
    reserve = evaluation_reserve(campaign.inputs, estimate)
    can_pair = len(slots) == 2 and len(pilots) == 2 and pilots[SCREEN_KEYS[0]]["trial"] is not None         and pilots[SCREEN_KEYS[0]]["trial"] == pilots[SCREEN_KEYS[1]]["trial"]
    rates = overlap_rate(*[pilots[key] for key in SCREEN_KEYS]) if can_pair else {"pair": False}
    stage_keys = SCREEN_KEYS if stage == "screen" else REPEAT_KEYS
    candidates = []
    for key in stage_keys:
        if status[key]["status"] == "complete":
            continue
        config = resolved_run(
            key, campaign=campaign.id, inputs=campaign.inputs, source=campaign.receipt["source"],
            runtime=runtime, physical_batch=physical_batch,
            device={"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}[runtime["backend"]],
            selection=snapshot["selection"],
        )
        candidates.append((key, run_seconds(
            estimate=estimate, config=config, steps=status[key]["steps"],
            completed_epochs=len(status[key]["epochs"]),
            selection_rows=len(campaign.inputs.rows["model_selection"]), screening=stage == "screen",
        )))
    chosen = choose_group(candidates, available, reserve, rates["pair"])
    future = []
    if stage == "screen":
        # The selected epoch is unknown. Budget both repeats at the three-epoch upper bound.
        base = {"steps_per_epoch": steps_per_epoch(len(campaign.inputs.rows["train"])),
                "max_epochs": 3}
        forecast_repeat = run_seconds(
            estimate=estimate, config=base, steps=0, completed_epochs=0,
            selection_rows=len(campaign.inputs.rows["model_selection"]), screening=False,
        )
        future = [forecast_repeat, forecast_repeat]
    current = [seconds for _, seconds in candidates]
    group_wall = max if rates["pair"] else sum
    all_work = [*current, *future]
    total_wall = sum(group_wall(group) for group in (current, future) if group)
    report = {
        "mode": stage, "forecast_seconds": dict(candidates), "admitted": list(chosen),
        "remaining_schedule_active_seconds": sum(all_work),
        "remaining_schedule_wall_seconds": total_wall,
        "remaining_schedule_fits": sum(all_work) + reserve <= min(available["active"], available["provider"])
        and total_wall <= available["wall"],
        "available_active_seconds": available["active"], "available_wall_seconds": available["wall"],
        "provider_available_seconds": available["provider"] if available["provider"] != float("inf") else None,
        "reserve": reserve, "timing": estimate, "rates": rates, "overlap": False,
    }
    # Cover the extra boundary persistence as the supervisor approaches its hard stop.
    grace = max(5, 2 * estimate["checkpoint_seconds"])
    return chosen, report, grace


def _write_summary(campaign, *, checkpoints=False, seal=False):
    result = campaign.inspect(checkpoints=checkpoints)
    path = campaign.directory / "development-summary.json"
    handoff = campaign.directory / "slice-8-handoff.json"
    if handoff.exists():
        frozen = read_json(path)
        digest = sha256_file(path)
        if (read_json(handoff)["development_summary_sha256"] != digest
                or {key: value for key, value in frozen.items() if key != "budget"} != result):
            raise ValueError("Sealed development evidence no longer matches saved artifacts.")
        return frozen, digest
    result["budget"] = budget_summary(campaign.artifacts)
    if seal and result["budget"]["unresolved_attempts"]:
        raise ValueError("Reconcile unresolved attempts before sealing Slice 6.")
    digest = save_json(path, result)
    if seal:
        save_json(handoff, {
            "campaign": campaign.id, "development_summary_sha256": digest,
            "selection": result["selection"], "full_schedule_complete": result["full_schedule_complete"],
            "scope": "joint_selection_pending" if result["full_schedule_complete"] else "lr_primary_transformer_exploratory",
            "complete_runs": [key for key, status in result["state"].items() if status["status"] == "complete"],
            "state": result["state"],
        })
    return result, digest


def run_stage(campaign, stage, *, backend, execution, physical_batch, privacy,
              quota_hours, session_hours, segment_hours):
    if (campaign.directory / "slice-8-handoff.json").exists():
        raise ValueError("This campaign has already been sealed for Slice 8.")
    campaign.start_inputs()
    ledger = budget_summary(campaign.artifacts)
    available = time_remaining(utc_now(), ledger, backend=backend, quota_hours=quota_hours,
                               session_hours=session_hours, segment_hours=segment_hours)
    runtime, slots = runtime_inventory(backend, user_privacy=privacy)
    if execution == "serial":
        slots = slots[:1]
    snapshot = campaign.inspect()
    trial = uuid4().hex
    chosen, forecast, grace = stage_plan(campaign, stage, snapshot, available, slots, runtime, physical_batch)
    forecast.update({"study_id": trial, "time": utc_now().isoformat(), "quota_before": quota_hours,
                     "picked": list(chosen), "session_hours": session_hours,
                     "execution": execution, "hardware_slots": list(slots)})
    save_json(campaign.state / "forecasts" / f"{trial}.json", forecast)
    if not chosen:
        return {"status": "no_admitted_work", "forecast": forecast,
                **export_campaign(campaign, campaign.directory / RETURN_NAME)}
    measured_bytes = forecast.get("timing", {}).get("checkpoint_bytes", 850_000_000)
    if shutil.disk_usage(campaign.directory).free < measured_bytes * (4 + len(chosen) * 3):
        raise ValueError("Insufficient free space for recoverable private checkpoints and return archives.")
    model_root = cached_model(campaign.artifacts.parent)
    available = time_remaining(utc_now(), budget_summary(campaign.artifacts), backend=backend,
                               quota_hours=quota_hours, session_hours=session_hours, segment_hours=segment_hours)
    seconds = allocation(available, len(chosen), forecast["reserve"], pilot=stage == "pilot")
    if seconds <= grace:
        raise ValueError("Insufficient allocation for a recoverable experiment segment.")
    device = {"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}[backend]
    jobs = []
    for index, key in enumerate(chosen):
        directory, config = campaign.stage_config(key, runtime, physical_batch, device, snapshot["selection"])
        jobs.append((key, directory, config, slots[index]))
    started = time.monotonic()
    outcomes = _supervise(campaign, jobs, stage=stage, model_root=model_root, seconds=seconds,
                          reserve_seconds=grace, overlap=forecast["overlap"], quota_hours=quota_hours,
                          trial=trial)
    forecast.update({"wall_seconds": time.monotonic() - started, "outcomes": outcomes})
    save_json(campaign.state / "forecasts" / f"{trial}.json", forecast)
    result, digest = _write_summary(campaign)
    archive = export_campaign(campaign, campaign.directory / RETURN_NAME)
    return {"status": "stage_returned", "outcomes": outcomes, "runs": result["state"],
            "forecast": forecast, "summary_sha256": digest, "budget": result["budget"], **archive}


def command(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(root / "artifacts"):
        if args.transformer_action == "prepare":
            return prepare_campaign(root, args.campaign_id, args.freeze, args.source_root or root)
        if args.transformer_action == "run" and args.input is not None:
            receive_campaign(root, args.campaign_id, args.input, args.receipt_sha256, args.source_root or root)
        campaign = Campaign(root, args.campaign_id, receipt_sha256=getattr(args, "receipt_sha256", None))
        if args.transformer_action == "collect":
            if (campaign.directory / "slice-8-handoff.json").exists():
                raise ValueError("This campaign has already been sealed for Slice 8.")
            campaign.start_inputs()
            import_campaign(campaign, args.results, expected_sha256=args.return_index_sha256)
        if args.transformer_action == "run":
            if args.results:
                campaign.start_inputs()
                import_campaign(campaign, args.results, expected_sha256=args.return_index_sha256)
            return run_stage(campaign, args.stage, backend=args.backend, execution=args.execution,
                             physical_batch=args.batch, privacy=args.privacy_confirmed,
                             quota_hours=args.gpu_hours_left, session_hours=args.session_hours_left,
                             segment_hours=args.segment_hours)
        result, digest = _write_summary(campaign, checkpoints=True, seal=getattr(args, "seal", False))
        return {"status": "verified", "state": result["state"], "selection": result["selection"],
                "full_schedule_complete": result["full_schedule_complete"],
                "summary_sha256": digest, "summary_path": str(campaign.directory / "development-summary.json")}
