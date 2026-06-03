#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Purple Llama – End-to-End Automation Script
============================================
Automates the full lifecycle for all Purple Llama components:
  1. Environment validation
  2. Dependency installation (per component)
  3. Running all tests
  4. Running CybersecurityBenchmarks (optional, requires an LLM endpoint)
  5. Generating a consolidated report

Usage
-----
Basic (tests only):
    python automate.py

With benchmark run (requires OpenAI-compatible LLM):
    python automate.py --run-benchmarks --llm-under-test OPENAI::gpt-4o --llm-api-key sk-...

Select specific components:
    python automate.py --components codeshield llamafirewall

Skip install (if already installed):
    python automate.py --skip-install

Output report to a custom path:
    python automate.py --report-path results/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("purple_llama_automation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()

COMPONENTS: Dict[str, dict] = {
    "codeshield": {
        "path": REPO_ROOT / "CodeShield",
        "install_cmd": [sys.executable, "-m", "pip", "install", "-e", "."],
        "test_cmd": [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "description": "Inference-time insecure code detection",
    },
    "cybersecuritybenchmarks": {
        "path": REPO_ROOT / "CybersecurityBenchmarks",
        "install_cmd": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ],
        "test_cmd": [
            sys.executable,
            "-m",
            "pytest",
            "benchmark/tests/",
            "-v",
            "--tb=short",
        ],
        "description": "LLM cybersecurity safety evaluation suite",
    },
    "llamafirewall": {
        "path": REPO_ROOT / "LlamaFirewall",
        "install_cmd": [sys.executable, "-m", "pip", "install", "-e", "."],
        "test_cmd": [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "description": "AI-centric security risk detection framework",
    },
    "sensitivedocclassification": {
        "path": REPO_ROOT / "SensitiveDocClassification",
        "install_cmd": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ],
        "test_cmd": [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "description": "Sensitive document classification",
    },
}

ALL_COMPONENT_KEYS = list(COMPONENTS.keys())

# Available CybersecurityBenchmarks benchmark types
BENCHMARK_TYPES = [
    "instruct",
    "autocomplete",
    "mitre",
    "prompt_injection",
    "interpreter",
]


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    status: str  # "passed" | "failed" | "skipped"
    duration_seconds: float
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class ComponentReport:
    component: str
    description: str
    install: Optional[StepResult] = None
    tests: Optional[StepResult] = None
    benchmark: Optional[StepResult] = None


@dataclass
class Report:
    generated_at: str
    python_version: str
    components: List[ComponentReport] = field(default_factory=list)
    overall_status: str = "unknown"

    def summary(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for cr in self.components:
            steps = [s for s in [cr.install, cr.tests, cr.benchmark] if s is not None]
            statuses = {s.status for s in steps}
            if "failed" in statuses:
                out[cr.component] = "FAILED"
            elif all(s == "passed" for s in statuses):
                out[cr.component] = "PASSED"
            else:
                out[cr.component] = "PARTIAL"
        return out


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def run_command(
    cmd: List[str],
    cwd: Path,
    step_name: str,
    timeout: int = 600,
) -> StepResult:
    """Run a subprocess command and return a StepResult."""
    LOG.info("  Running: %s", " ".join(str(c) for c in cmd))
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        status = "passed" if proc.returncode == 0 else "failed"
        if status == "failed":
            LOG.warning("  Step '%s' failed (rc=%d)", step_name, proc.returncode)
            if proc.stdout:
                LOG.warning("  stdout:\n%s", proc.stdout[-2000:])
            if proc.stderr:
                LOG.warning("  stderr:\n%s", proc.stderr[-2000:])
        return StepResult(
            name=step_name,
            status=status,
            duration_seconds=round(duration, 2),
            return_code=proc.returncode,
            stdout=proc.stdout[-4000:],  # cap output size
            stderr=proc.stderr[-4000:],
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        LOG.error("  Step '%s' timed out after %ds", step_name, timeout)
        return StepResult(
            name=step_name,
            status="failed",
            duration_seconds=round(duration, 2),
            return_code=-1,
            stderr=f"Timed out after {timeout}s",
        )
    except Exception as exc:
        duration = time.monotonic() - start
        LOG.error("  Step '%s' raised an exception: %s", step_name, exc)
        return StepResult(
            name=step_name,
            status="failed",
            duration_seconds=round(duration, 2),
            return_code=-1,
            stderr=str(exc),
        )


def check_python_version() -> None:
    """Ensure Python >= 3.9."""
    if sys.version_info < (3, 9):
        LOG.error(
            "Python 3.9+ required. Current version: %s", sys.version
        )
        sys.exit(1)
    LOG.info("Python version: %s", sys.version.split()[0])


def check_pip_available() -> None:
    """Ensure pip is available."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        LOG.error("pip is not available. Please install pip.")
        sys.exit(1)
    LOG.info("pip: %s", result.stdout.strip())


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------
def install_component(component_key: str, info: dict) -> StepResult:
    LOG.info("[%s] Installing dependencies...", component_key)
    return run_command(
        cmd=info["install_cmd"],
        cwd=info["path"],
        step_name=f"{component_key}_install",
        timeout=300,
    )


def test_component(component_key: str, info: dict) -> StepResult:
    LOG.info("[%s] Running tests...", component_key)
    return run_command(
        cmd=info["test_cmd"],
        cwd=info["path"],
        step_name=f"{component_key}_tests",
        timeout=300,
    )


def run_benchmark(
    benchmark_type: str,
    llm_under_test: str,
    llm_api_key: str,
    num_queries: int,
    output_dir: Path,
) -> StepResult:
    """Run a single CybersecurityBenchmarks benchmark type."""
    bench_path = REPO_ROOT / "CybersecurityBenchmarks"
    prompt_path = bench_path / "datasets" / benchmark_type
    response_path = output_dir / f"{benchmark_type}_responses.json"
    stat_path = output_dir / f"{benchmark_type}_stats.json"

    # Determine the dataset file (first json found)
    json_files = list(prompt_path.glob("*.json")) if prompt_path.exists() else []
    if not json_files:
        LOG.warning(
            "[benchmark:%s] No dataset JSON found in %s, skipping.",
            benchmark_type,
            prompt_path,
        )
        return StepResult(
            name=f"benchmark_{benchmark_type}",
            status="skipped",
            duration_seconds=0.0,
            stderr="No dataset file found",
        )

    # Set API key in environment (do not log the key value)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = llm_api_key

    cmd = [
        sys.executable,
        "-m",
        "CybersecurityBenchmarks.benchmark.run",
        "--benchmark-type",
        benchmark_type,
        "--prompt-path",
        str(json_files[0]),
        "--response-path",
        str(response_path),
        "--stat-path",
        str(stat_path),
        "--llm-under-test",
        llm_under_test,
        "--num-queries-per-prompt",
        str(num_queries),
    ]

    LOG.info("[benchmark:%s] Running benchmark...", benchmark_type)
    return run_command(
        cmd=cmd,
        cwd=REPO_ROOT,
        step_name=f"benchmark_{benchmark_type}",
        timeout=1800,  # 30 minutes per benchmark type
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(report: Report, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.summary()

    # Determine overall status
    statuses = set(summary.values())
    if "FAILED" in statuses:
        report.overall_status = "FAILED"
    elif statuses == {"PASSED"}:
        report.overall_status = "PASSED"
    else:
        report.overall_status = "PARTIAL"

    # Write JSON report
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)

    # Print human-readable summary to console
    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("PURPLE LLAMA AUTOMATION REPORT")
    LOG.info("=" * 60)
    for component, status in summary.items():
        icon = "✓" if status == "PASSED" else ("✗" if status == "FAILED" else "~")
        LOG.info("  %s  %-35s %s", icon, component, status)
    LOG.info("-" * 60)
    LOG.info("  Overall status: %s", report.overall_status)
    LOG.info("  Full report:    %s", report_path)
    LOG.info("=" * 60)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Purple Llama – End-to-End Automation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=ALL_COMPONENT_KEYS + ["all"],
        default=["all"],
        help="Components to process (default: all)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation steps",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test execution steps",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run CybersecurityBenchmarks against an LLM (requires --llm-under-test and --llm-api-key)",
    )
    parser.add_argument(
        "--benchmark-types",
        nargs="+",
        choices=BENCHMARK_TYPES,
        default=["instruct", "mitre"],
        help="Which benchmark suites to run (default: instruct, mitre)",
    )
    parser.add_argument(
        "--llm-under-test",
        type=str,
        default=None,
        help="LLM identifier, e.g. OPENAI::gpt-4o or TOGETHER::meta-llama/Llama-3-70b-chat-hf",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for the LLM provider (or set via OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of queries per prompt in benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "automation_report.json",
        help="Path for the output JSON report",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_outputs",
        help="Directory for benchmark response/stat files",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # --- Resolve components ---
    if "all" in args.components:
        selected_keys = ALL_COMPONENT_KEYS
    else:
        selected_keys = args.components

    # --- Pre-flight checks ---
    check_python_version()
    check_pip_available()

    # --- Benchmark pre-flight ---
    if args.run_benchmarks:
        api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not args.llm_under_test:
            LOG.error(
                "--llm-under-test is required when --run-benchmarks is set."
            )
            return 1
        if not api_key:
            LOG.error(
                "An API key is required. Pass --llm-api-key or set OPENAI_API_KEY."
            )
            return 1
    else:
        api_key = ""

    report = Report(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version.split()[0],
    )

    # --- Per-component pipeline ---
    for key in selected_keys:
        info = COMPONENTS[key]
        comp_path: Path = info["path"]

        if not comp_path.exists():
            LOG.warning("[%s] Directory not found: %s — skipping.", key, comp_path)
            continue

        LOG.info("")
        LOG.info("━" * 60)
        LOG.info(" Component: %s", key.upper())
        LOG.info(" %s", info["description"])
        LOG.info("━" * 60)

        cr = ComponentReport(component=key, description=info["description"])

        # Install
        if not args.skip_install:
            cr.install = install_component(key, info)
            if cr.install.status == "failed":
                LOG.warning(
                    "[%s] Install failed; tests may also fail.", key
                )
        else:
            cr.install = StepResult(
                name=f"{key}_install",
                status="skipped",
                duration_seconds=0.0,
            )

        # Tests
        if not args.skip_tests:
            cr.tests = test_component(key, info)
        else:
            cr.tests = StepResult(
                name=f"{key}_tests",
                status="skipped",
                duration_seconds=0.0,
            )

        report.components.append(cr)

    # --- Benchmarks (CybersecurityBenchmarks only) ---
    if args.run_benchmarks:
        LOG.info("")
        LOG.info("━" * 60)
        LOG.info(" Running CybersecurityBenchmarks")
        LOG.info("━" * 60)
        args.benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        # Find or create the cybersecuritybenchmarks component report
        bench_cr = next(
            (c for c in report.components if c.component == "cybersecuritybenchmarks"),
            None,
        )
        if bench_cr is None:
            bench_info = COMPONENTS["cybersecuritybenchmarks"]
            bench_cr = ComponentReport(
                component="cybersecuritybenchmarks",
                description=bench_info["description"],
                install=StepResult(
                    name="cybersecuritybenchmarks_install",
                    status="skipped",
                    duration_seconds=0.0,
                ),
                tests=StepResult(
                    name="cybersecuritybenchmarks_tests",
                    status="skipped",
                    duration_seconds=0.0,
                ),
            )
            report.components.append(bench_cr)

        # Run each requested benchmark type; store combined result
        bench_results: List[StepResult] = []
        for btype in args.benchmark_types:
            result = run_benchmark(
                benchmark_type=btype,
                llm_under_test=args.llm_under_test,  # type: ignore[arg-type]
                llm_api_key=api_key,
                num_queries=args.num_queries,
                output_dir=args.benchmark_output_dir,
            )
            bench_results.append(result)

        # Aggregate benchmark results
        all_passed = all(r.status == "passed" for r in bench_results)
        any_failed = any(r.status == "failed" for r in bench_results)
        combined_status = "passed" if all_passed else ("failed" if any_failed else "skipped")
        bench_cr.benchmark = StepResult(
            name="benchmark_combined",
            status=combined_status,
            duration_seconds=round(sum(r.duration_seconds for r in bench_results), 2),
            stdout="\n---\n".join(r.stdout for r in bench_results),
            stderr="\n---\n".join(r.stderr for r in bench_results),
        )

    # --- Write report ---
    generate_report(report, args.report_path)

    # Return non-zero exit code if anything failed
    final_status = report.overall_status
    return 0 if final_status in ("PASSED", "PARTIAL") else 1


if __name__ == "__main__":
    sys.exit(main())
