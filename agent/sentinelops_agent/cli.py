import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import joblib

from sentinelops_agent.model_check import _resolve_model_path, run_model_check
from sentinelops_agent.monitoring import data_quality, drift_report, monitoring_report
from sentinelops_agent.scanner import scan_project, discover_monitoring_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [agent] %(message)s")
logger = logging.getLogger(__name__)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def post(api_url: str, path: str, token: str, payload: dict):
    url = f"{api_url.rstrip('/')}{path}"
    logger.info("POST %s", url)
    try:
        response = requests.post(url, json=payload, headers=auth_headers(token), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection failed: unable to connect to backend at %s. Is the FastAPI server running?", api_url)
        raise SystemExit(1) from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 401:
            logger.error("Authentication failed (HTTP 401): Invalid or revoked Agent credential token.")
        elif status_code == 403:
            logger.error("Access forbidden (HTTP 403): Token lacks 'model:check' scope.")
        elif status_code == 404:
            logger.error("Resource not found (HTTP 404) at %s: Please verify your model-id or endpoint.", path)
        else:
            logger.error("HTTP error %s: %s", status_code, exc.response.text if exc.response is not None else exc)
        raise SystemExit(1) from exc


def get(api_url: str, path: str, token: str) -> dict:
    url = f"{api_url.rstrip('/')}{path}"
    logger.info("GET %s", url)
    try:
        response = requests.get(url, headers=auth_headers(token), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        logger.error("Connection failed: unable to connect to backend at %s. Is the FastAPI server running?", api_url)
        raise SystemExit(1) from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 401:
            logger.error("Authentication failed (HTTP 401): Invalid or revoked Agent credential token.")
        elif status_code == 403:
            logger.error("Access forbidden (HTTP 403): Token lacks 'model:check' scope.")
        elif status_code == 404:
            logger.error("Resource not found (HTTP 404) at %s: Please verify your model-id or endpoint.", path)
        else:
            logger.error("HTTP error %s: %s", status_code, exc.response.text if exc.response is not None else exc)
        raise SystemExit(1) from exc


def submit_check(api_url: str, model_id: str, token: str, result: dict) -> dict:
    body = post(api_url, f"/agent/models/{model_id}/check", token, result)
    print(json.dumps(body, indent=2))
    return body


def run_local_check_and_submit(
    api_url: str,
    model_id: str,
    token: str,
    *,
    model_path: str | None = None,
    framework: str | None = None,
    model_type: str | None = None,
    sample_path: str | None = None,
) -> dict:
    if not model_path or not framework or not model_type:
        config = get(api_url, f"/agent/models/{model_id}", token)
        model_path = model_path or config.get("local_model_path")
        framework = framework or config.get("framework") or "scikit-learn"
        model_type = model_type or config.get("model_type") or "classification"
    if not model_path:
        raise SystemExit("No local_model_path configured for this model.")
    logger.info("Running model check model_id=%s path=%s", model_id, model_path)
    result = run_model_check(
        model_id=model_id,
        model_path=model_path,
        framework=framework,
        model_type=model_type,
        sample_features_path=sample_path,
    )
    return submit_check(api_url, model_id, token, result)


def run_auto_monitoring_and_submit(api_url: str, model_id: str, token: str, model_path: str) -> dict:
    try:
        model_file = _resolve_model_path(model_path)
        model = joblib.load(model_file)
        model_features = [str(x) for x in getattr(model, "feature_names_in_", [])] or None
        detection = discover_monitoring_inputs(str(model_file), model_features)
        logger.info("Auto-detected monitoring data reference=%s current=%s label=%s", detection["reference_path"], detection["current_path"], detection.get("label_column"))
        reference = pd.read_csv(detection["reference_path"])
        current = pd.read_csv(detection["current_path"])
        payload = monitoring_report(reference, current, model, detection.get("label_column"))
        return post(api_url, f"/agent/models/{model_id}/monitoring", token, payload)
    except Exception as exc:
        logger.error("Automatic monitoring failed: %s", exc)
        raise SystemExit(1) from exc

def serve_loop(api_url: str, model_id: str, token: str, interval: float) -> None:
    logger.info("Agent serve started model_id=%s interval=%ss", model_id, interval)
    while True:
        try:
            work = get(api_url, f"/agent/models/{model_id}/work", token)
            if work.get("action") == "model_check":
                run_local_check_and_submit(
                    api_url,
                    model_id,
                    token,
                    model_path=work.get("local_model_path"),
                    framework=work.get("framework"),
                    model_type=work.get("model_type"),
                )
            elif work.get("action") == "monitor_model":
                run_auto_monitoring_and_submit(
                    api_url,
                    model_id,
                    token,
                    model_path=work.get("local_model_path"),
                )
        except SystemExit:
            # get()/post() already logged a clear reason; keep the daemon alive and retry.
            logger.warning("Agent request failed; retrying in %ss", interval)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent loop error: %s", exc)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="SentinelOps local Agent")
    subs = parser.add_subparsers(dest="command", required=True)

    check = subs.add_parser("check", help="Run a one-shot local model check and upload results")
    check.add_argument("--api-url", default="http://localhost:8000")
    check.add_argument("--model-id", required=True)
    check.add_argument("--token", required=True)
    check.add_argument("--model-path")
    check.add_argument("--sample-path")

    serve = subs.add_parser("serve", help="Poll backend for pending model checks")
    serve.add_argument("--api-url", default="http://localhost:8000")
    serve.add_argument("--model-id", required=True)
    serve.add_argument("--token", required=True)
    serve.add_argument("--interval", type=float, default=2.0)

    for command in ("scan", "monitor"):
        p = subs.add_parser(command)
        p.add_argument("--project-root", required=True)
        p.add_argument("--api-url", default="http://localhost:8000")
        p.add_argument("--project-id", required=True)
        p.add_argument("--token", required=True)
    subs.choices["monitor"].add_argument("--reference", required=True)
    subs.choices["monitor"].add_argument("--current", required=True)

    monitor_model = subs.add_parser("monitor-model", help="Run local monitoring for a registered model and upload aggregate results")
    monitor_model.add_argument("--api-url", default="http://localhost:8000")
    monitor_model.add_argument("--model-id", required=True)
    monitor_model.add_argument("--token", required=True)
    monitor_model.add_argument("--reference", required=True)
    monitor_model.add_argument("--current", required=True)
    monitor_model.add_argument("--label-column", help="Optional ground-truth label column in the current CSV")

    args = parser.parse_args()

    if args.command == "check":
        run_local_check_and_submit(
            args.api_url,
            args.model_id,
            args.token,
            model_path=args.model_path,
            sample_path=args.sample_path,
        )
        return

    if args.command == "serve":
        serve_loop(args.api_url, args.model_id, args.token, args.interval)
        return

    if args.command == "scan":
        post(
            args.api_url,
            f"/agent/projects/{args.project_id}/scan",
            args.token,
            {"project_root": str(Path(args.project_root).resolve()), "findings": scan_project(args.project_root)},
        )
        return

    if args.command == "monitor-model":
        reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
        config = get(args.api_url, f"/agent/models/{args.model_id}", args.token)
        model_path = config.get("local_model_path")
        if not model_path:
            raise SystemExit("No local_model_path configured for this model.")
        try:
            model = joblib.load(_resolve_model_path(model_path))
        except Exception as exc:  # retain drift/quality reporting if only performance load fails
            logger.warning("Could not load model for performance metrics: %s", exc)
            model = None
        payload = monitoring_report(reference, current, model, args.label_column)
        post(args.api_url, f"/agent/models/{args.model_id}/monitoring", args.token, payload)
        return

    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    post(
        args.api_url,
        f"/agent/projects/{args.project_id}/monitoring",
        args.token,
        {"data_quality": data_quality(cur), "drift": drift_report(ref, cur)},
    )


if __name__ == "__main__":
    main()
