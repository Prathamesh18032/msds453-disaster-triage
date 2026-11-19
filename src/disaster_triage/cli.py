"""Thin CLI for the local experiment workflow."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog="disaster-triage")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "smoke", "budget-summary"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--config", type=Path)
        command.add_argument("--device")
        if name == "smoke":
            command.add_argument("--stage", choices=("offline", "pretrained", "resume"), default="offline")
            command.add_argument("--source-root", type=Path, default=Path.cwd())
            command.add_argument("--resume-dir", type=Path)
    data = sub.add_parser("data", help="local dataset preparation; human audit required")
    actions = data.add_subparsers(dest="data_action", required=True)
    for name in ("inspect", "prepare-audit", "freeze", "verify"):
        command = actions.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name in ("prepare-audit", "freeze"):
            command.add_argument("--source-root", type=Path)
        if name == "freeze":
            command.add_argument("--decisions", type=Path, required=True)
        if name == "verify":
            command.add_argument("--freeze", type=Path, required=True)
    classical = sub.add_parser("classical", help="local CPU classical model selection")
    classical_actions = classical.add_subparsers(dest="classical_action", required=True)
    for name in ("run", "verify"):
        command = classical_actions.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--freeze", type=Path, required=True)
        command.add_argument("--campaign-id", required=True)
        if name == "run":
            command.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "classical":
        from .classical import ClassicalError
        from .classical_run import DeadlineReached, run_classical
        try:
            result = run_classical(
                args.root, args.freeze, args.campaign_id,
                source_root=getattr(args, "source_root", None),
                verify=args.classical_action == "verify",
            )
        except DeadlineReached:
            print("Classical execution reached the experiment cap or cutoff.", file=sys.stderr)
            return 130
        except KeyboardInterrupt:
            print("Classical execution interrupted.", file=sys.stderr)
            return 130
        except ClassicalError as error:
            print(str(error), file=sys.stderr)
            return 1
        except (ValueError, RuntimeError, OSError, TypeError, KeyError, EOFError, ImportError) as error:
            print("Classical execution failed (" + type(error).__name__ +
                  "). Check the freeze, pinned runtime, private artifacts, and budget.", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "data":
        import csv
        import tarfile
        from .dataset import inspect_data, prepare_audit, freeze_data, verify_freeze
        try:
            if args.data_action == "inspect":
                result = inspect_data(args.root)
            elif args.data_action == "prepare-audit":
                result = prepare_audit(args.root, args.source_root or args.root)
            elif args.data_action == "freeze":
                result = freeze_data(args.root, args.source_root or args.root, args.decisions)
            else:
                result = verify_freeze(args.root, args.freeze)
            print(json.dumps(result, indent=2))
            return 0
        except (ValueError, RuntimeError, OSError, TypeError, KeyError, EOFError,
                csv.Error, tarfile.TarError) as error:
            # Never echo raw parse values, source IDs, notes, or remote exceptions.
            print("Data preparation failed (" + type(error).__name__ +
                  "). Check archive identity, complete audit decisions, and local data paths.", file=sys.stderr)
            return 1
    from .config import RunConfig, load_config
    config = load_config(args.config) if args.config else RunConfig()
    if args.device:
        config = replace(config, device=args.device)
    try:
        if args.command == "doctor":
            from .runtime import doctor
            result = doctor(config, args.root)
            print(json.dumps(result, indent=2))
            return 0 if result["python_supported"] and result["requested_device_available"] and not result["profile_errors"] and all(result["versions"].values()) else 1
        if args.command == "budget-summary":
            from .accounting import budget_summary
            result = budget_summary(config.paths(args.root)["artifact_dir"])
        else:
            from .smoke import run_offline, run_pretrained, resume_probe
            if args.stage == "resume":
                if args.resume_dir is None:
                    parser.error("--resume-dir is required for resume")
                result = resume_probe(config, args.root, args.source_root, args.resume_dir)
            else:
                result = (run_offline if args.stage == "offline" else run_pretrained)(
                    config, args.root, args.source_root,
                )
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        # Library diagnostics contain metadata only. Avoid arbitrary remote/API messages.
        print("Execution failed (" + type(error).__name__ + "). Check the selected profile and private logs.",
              file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
