# Private L4 Benchmark Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-testable, fail-closed L4 handoff that runs the reviewed source against the immutable paired A100 artifacts, benchmarks PyTorch FP32, ONNX Runtime CUDA FP32, and TensorRT FP16, and produces a private verified package without retraining.

**Architecture:** Add one shared L4 identity/verification module, then make the benchmark, package builder, and stage-specific handoff consume that contract. The L4 notebook remains a thin orchestration layer: it verifies the rendered source bundle, installs the locked environment, calls the tested modules, and prints success only after the report and ZIP sidecar verify. Existing A100 and parity-probe behavior remains unchanged.

**Tech Stack:** Python 3.11, `dataclasses`, `argparse`, Git bundles, JSON/YAML, pytest, Ultralytics 8.4.89, PyTorch CUDA 12.6, ONNX Runtime GPU 1.26.0, TensorRT supplied by the Colab L4 runtime, `uv`, Ruff.

## Global Constraints

- This local implementation phase is CPU-only: do not train, export a real model, invoke CUDA, or run a GPU benchmark.
- Do not read `.env`; do not access another repository or the parent directory.
- Do not alter Git history, remotes, branches, tags, releases, GitHub, Hugging Face, or official accounts.
- Preserve the existing A100 experiment and parity-probe paths; do not overwrite their Drive directories or local handoffs.
- Parent experiment Git SHA: `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`.
- Raw deployment-gate SHA-256: `466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803`.
- Selected checkpoint SHA-256: `44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a`.
- ONNX SHA-256: `b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`.
- Dataset SHA-256: `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`.
- Protocol-manifest SHA-256: `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`.
- Parent workspace: `/content/drive/MyDrive/pcb-defect-paired/workspaces/9e3a1ed5827a`; never derive it from the runner SHA.
- Benchmark only the frozen 60-image `calibration` partition; final-test images, INT8, batching, power, SLA, and production claims remain out of scope.
- TensorRT is a measured system dependency: reject absence, record its exact version, and never auto-install it.
- Result ZIP, sidecar, checkpoint, ONNX, TensorRT engine, dataset pixels, logs, and generated handoff stay ignored/untracked and private.
- Repository code remains `AGPL-3.0-or-later`; this private run does not establish dataset/model redistribution rights or authorize commercial use.
- Do not change README, claims, model card, or release status in this phase; public metadata promotion begins only after a separate returned-package audit.
- Every implementation task follows red-green TDD and ends with its own commit using `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` and no co-author.
- Estimated execution: 45–90 minutes for local CPU hardening/review/handoff generation, then 15–40 minutes for the separate user-operated Colab L4 run.

## File Structure

- Create `src/pcb_defect/l4_contract.py`: standard-library-only immutable runner/parent identity types and CPU-safe verification of Git, gate, checkpoint, ONNX, protocol, and calibration inputs, callable before project installation.
- Create `src/pcb_defect/l4_package.py`: collect the private L4 evidence set, derive the two-identity filename, and reuse deterministic ZIP creation/verification.
- Create `src/pcb_defect/l4_handoff.py`: render one stage-specific notebook and manifest into a runner-prefixed immutable handoff directory.
- Modify `src/pcb_defect/benchmark.py`: consume the shared contract, record both provenances, enforce L4/TensorRT/runtime gates, preserve complete statistics, and verify resumability.
- Modify `src/pcb_defect/result_package.py`: add a reusable strict verifier for an existing ZIP plus `.sha256` pair.
- Modify `src/pcb_defect/handoff.py`: expose the existing project metadata and notebook renderer as reusable internal package interfaces without changing the existing CLI result.
- Modify `notebooks/deployment_benchmark_l4.ipynb`: separate runner and parent identities and orchestrate only verify, benchmark, package, and final verification steps.
- Create `tests/test_l4_contract.py`, `tests/test_l4_package.py`, and `tests/test_l4_handoff.py`.
- Modify `tests/test_benchmark.py`, `tests/test_result_package.py`, `tests/test_handoff.py`, and `tests/test_release_contract.py`.

---

### Task 1: Shared L4 Identity and Parent Verification Contract

**Files:**
- Create: `src/pcb_defect/l4_contract.py`
- Create: `tests/test_l4_contract.py`

**Interfaces:**
- Consumes: a checked-out runner repository, the immutable parent workspace, raw deployment-gate JSON, `inputs/paired_split_manifest.json`, `deployment/calibration.yaml`, the checkpoint, ONNX, and calibration images.
- Produces: `L4ContractError`; `L4ParentIdentity.parse(*, experiment_git_sha: str, deployment_gate_sha256: str, checkpoint_sha256: str, onnx_sha256: str) -> L4ParentIdentity`; `L4RunIdentity.parse(*, runner_git_sha: str, experiment_git_sha: str, deployment_gate_sha256: str, checkpoint_sha256: str, onnx_sha256: str) -> L4RunIdentity`; `VerifiedL4ParentInputs`; `VerifiedL4Inputs`; `verify_l4_parent_inputs(workspace: Path, parent: L4ParentIdentity) -> VerifiedL4ParentInputs`; and `verify_l4_inputs(repo: Path, workspace: Path, identity: L4RunIdentity) -> VerifiedL4Inputs`.
- Test-only helpers defined in `tests/test_l4_contract.py`: `_sha256_json(value: Any) -> str`; `_write_clean_runner_repo(tmp_path: Path) -> tuple[Path, str]`, which creates one clean commit and returns its SHA; `_write_parent_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]`; `_valid_contract_fixture(tmp_path: Path) -> tuple[Path, Path, L4RunIdentity]`; and `_apply_mutation(repo: Path, workspace: Path, mutation: str) -> None` for byte-only fail-closed cases.

- [ ] **Step 1: Write identity parsing tests that require exact lowercase hexadecimal values and distinct runner/parent SHAs**

```python
from pcb_defect.l4_contract import L4ContractError, L4RunIdentity


def test_l4_identity_requires_distinct_runner_and_parent() -> None:
    with pytest.raises(L4ContractError, match="runner and parent experiment Git SHAs must differ"):
        L4RunIdentity.parse(
            runner_git_sha="a" * 40,
            experiment_git_sha="a" * 40,
            deployment_gate_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            onnx_sha256="d" * 64,
        )


@pytest.mark.parametrize("bad", ["", "A" * 64, "g" * 64, "a" * 63])
def test_l4_identity_rejects_malformed_sha256(bad: str) -> None:
    with pytest.raises(L4ContractError, match="64 lowercase hexadecimal"):
        L4RunIdentity.parse(
            runner_git_sha="a" * 40,
            experiment_git_sha="b" * 40,
            deployment_gate_sha256=bad,
            checkpoint_sha256="c" * 64,
            onnx_sha256="d" * 64,
        )
```

- [ ] **Step 2: Run the identity tests and confirm the missing module failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_contract.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'pcb_defect.l4_contract'`.

- [ ] **Step 3: Implement the immutable identity types and exact parser**

```python
@dataclass(frozen=True, slots=True)
class L4ParentIdentity:
    experiment_git_sha: str
    deployment_gate_sha256: str
    checkpoint_sha256: str
    onnx_sha256: str

    @classmethod
    def parse(
        cls,
        *,
        experiment_git_sha: str,
        deployment_gate_sha256: str,
        checkpoint_sha256: str,
        onnx_sha256: str,
    ) -> "L4ParentIdentity":
        _require_lowercase_hex(experiment_git_sha, 40, "parent experiment Git SHA")
        _require_lowercase_hex(deployment_gate_sha256, 64, "deployment-gate SHA-256")
        _require_lowercase_hex(checkpoint_sha256, 64, "checkpoint SHA-256")
        _require_lowercase_hex(onnx_sha256, 64, "ONNX SHA-256")
        return cls(experiment_git_sha, deployment_gate_sha256, checkpoint_sha256, onnx_sha256)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class L4RunIdentity:
    runner_git_sha: str
    parent: L4ParentIdentity

    @classmethod
    def parse(cls, *, runner_git_sha: str, **parent_values: str) -> "L4RunIdentity":
        _require_lowercase_hex(runner_git_sha, 40, "runner Git SHA")
        parent = L4ParentIdentity.parse(**parent_values)
        if runner_git_sha == parent.experiment_git_sha:
            raise L4ContractError("runner and parent experiment Git SHAs must differ")
        return cls(runner_git_sha, parent)
```

- [ ] **Step 4: Run the identity tests and confirm they pass**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Add a CPU fixture that creates a one-commit runner and a minimal content-addressed parent workspace**

```python
def _write_parent_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / ("b" * 12)
    deployment = workspace / "deployment"
    inputs = workspace / "inputs"
    runtime = workspace / "runtime_data" / "grouped"
    images = workspace / "dataset" / "images"
    for directory in (deployment, inputs, runtime, images):
        directory.mkdir(parents=True, exist_ok=True)
    calibration = images / "01_missing_hole_02.jpg"
    calibration.write_bytes(b"calibration-image")
    calibration_list = runtime / "calibration.txt"
    calibration_list.write_text(f"{calibration}\n", encoding="utf-8")
    (deployment / "calibration.yaml").write_text(
        f"train: {calibration_list}\nval: {calibration_list}\ntest: {calibration_list}\n",
        encoding="utf-8",
    )
    checkpoint = workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    onnx = deployment / "best.onnx"
    onnx.write_bytes(b"onnx")
    sample = {
        "stem": calibration.stem,
        "board_id": "01",
        "class_name": "missing_hole",
        "image_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "label_sha256": "0" * 64,
    }
    dataset_sha256 = _sha256_json([sample])
    manifest_payload = {
        "protocol_version": "paired-board-sensitivity-v1",
        "config": {},
        "dataset": {"sample_count": 1, "sha256": dataset_sha256, "samples": [sample]},
        "board_roles": {
            "final_test_and_exposure": "08",
            "validation_and_calibration": "01",
            "grouped_train": [],
        },
        "partitions": {"calibration": [calibration.stem], "final_test": []},
        "counts": {},
    }
    manifest_sha256 = _sha256_json(manifest_payload)
    (inputs / "paired_split_manifest.json").write_text(
        json.dumps({**manifest_payload, "manifest_sha256": manifest_sha256}),
        encoding="utf-8",
    )
    experiment_git_sha = "b" * 40
    gate = {
        "passed": True,
        "git_sha": experiment_git_sha,
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "artifacts": {
            "source_checkpoint": checkpoint.relative_to(workspace).as_posix(),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx": "best.onnx",
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        },
        "fidelity": {"pt": {"map50_95": 0.12}, "onnx": {"map50_95": 0.11}, "threshold": 0.02},
    }
    gate_path = deployment / "deployment_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return workspace, {
        "deployment_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }
```

Define the test-only `_sha256_json` with the production canonical JSON rule: UTF-8, `ensure_ascii=False`, `sort_keys=True`, and separators `(',', ':')`.

- [ ] **Step 6: Write the successful parent verification test**

```python
def test_verify_l4_inputs_binds_runner_parent_and_calibration(tmp_path: Path) -> None:
    repo, runner_sha = _write_clean_runner_repo(tmp_path)
    workspace, values = _write_parent_workspace(tmp_path)
    identity = L4RunIdentity.parse(
        runner_git_sha=runner_sha,
        experiment_git_sha="b" * 40,
        deployment_gate_sha256=values["deployment_gate_sha256"],
        checkpoint_sha256=values["checkpoint_sha256"],
        onnx_sha256=values["onnx_sha256"],
    )

    verified = verify_l4_inputs(repo, workspace, identity)

    assert verified.runner_git_sha == runner_sha
    assert verified.parent.experiment_git_sha == "b" * 40
    assert [path.stem for path in verified.parent.calibration_images] == [
        "01_missing_hole_02"
    ]
    assert verified.parent.checkpoint_path.is_relative_to(workspace)
    assert verified.parent.onnx_path.is_relative_to(workspace)
```

- [ ] **Step 7: Run the successful verification test and confirm it fails because `verify_l4_inputs` is absent**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_contract.py::test_verify_l4_inputs_binds_runner_parent_and_calibration -q`

Expected: FAIL because `verify_l4_inputs` or `VerifiedL4Inputs` is not defined.

- [ ] **Step 8: Implement `VerifiedL4Inputs` and fail-closed verification**

```python
@dataclass(frozen=True, slots=True)
class VerifiedL4ParentInputs:
    experiment_git_sha: str
    gate_path: Path
    gate: dict[str, Any]
    checkpoint_path: Path
    onnx_path: Path
    calibration_yaml: Path
    calibration_images: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class VerifiedL4Inputs:
    runner_git_sha: str
    parent: VerifiedL4ParentInputs


def verify_l4_parent_inputs(
    workspace: Path, parent: L4ParentIdentity
) -> VerifiedL4ParentInputs:
    workspace = workspace.resolve()
    if workspace.name != parent.experiment_git_sha[:12]:
        raise L4ContractError("parent workspace is not derived from the parent experiment SHA")
    gate_path = workspace / "deployment" / "deployment_gate.json"
    if _sha256_file(gate_path) != parent.deployment_gate_sha256:
        raise L4ContractError("raw deployment-gate SHA-256 mismatch")
    gate = _read_json_object(gate_path, "deployment gate")
    if gate.get("passed") is not True or gate.get("git_sha") != parent.experiment_git_sha:
        raise L4ContractError("deployment gate did not pass for the expected parent experiment")
    manifest_path = workspace / "inputs" / "paired_split_manifest.json"
    manifest = _read_json_object(manifest_path, "protocol manifest")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _sha256_json(manifest_payload) != manifest.get("manifest_sha256"):
        raise L4ContractError("protocol-manifest SHA-256 mismatch")
    if gate["dataset_sha256"] != manifest["dataset"]["sha256"]:
        raise L4ContractError("dataset SHA-256 mismatch")
    if gate["manifest_sha256"] != manifest["manifest_sha256"]:
        raise L4ContractError("protocol-manifest identity mismatch")
    checkpoint_path = _contained_path(
        workspace, workspace / gate["artifacts"]["source_checkpoint"], "checkpoint"
    )
    onnx_path = _contained_path(
        workspace, workspace / "deployment" / gate["artifacts"]["onnx"], "ONNX"
    )
    if _sha256_file(checkpoint_path) != parent.checkpoint_sha256:
        raise L4ContractError("checkpoint SHA-256 mismatch")
    if _sha256_file(onnx_path) != parent.onnx_sha256:
        raise L4ContractError("ONNX SHA-256 mismatch")
    calibration_yaml = workspace / "deployment" / "calibration.yaml"
    calibration_config = _read_simple_calibration_yaml(calibration_yaml)
    expected_list = (workspace / "runtime_data" / "grouped" / "calibration.txt").resolve()
    if Path(calibration_config["val"]).resolve() != expected_list:
        raise L4ContractError("calibration list path mismatch")
    calibration_images = tuple(
        Path(line).resolve()
        for line in expected_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    observed_stems = [path.stem for path in calibration_images]
    expected_stems = manifest["partitions"]["calibration"]
    if observed_stems != expected_stems or set(observed_stems) & set(
        manifest["partitions"]["final_test"]
    ):
        raise L4ContractError("calibration partition mismatch")
    sample_by_stem = {row["stem"]: row for row in manifest["dataset"]["samples"]}
    for path in calibration_images:
        if _sha256_file(path) != sample_by_stem[path.stem]["image_sha256"]:
            raise L4ContractError("calibration image SHA-256 mismatch")
    return VerifiedL4ParentInputs(
        parent.experiment_git_sha,
        gate_path,
        gate,
        checkpoint_path,
        onnx_path,
        calibration_yaml,
        calibration_images,
    )


def verify_l4_inputs(repo: Path, workspace: Path, identity: L4RunIdentity) -> VerifiedL4Inputs:
    repo = repo.resolve()
    observed_runner = _git(repo, "rev-parse", "HEAD")
    if observed_runner != identity.runner_git_sha or _git(repo, "status", "--porcelain"):
        raise L4ContractError("runner repository identity or cleanliness mismatch")
    parent = verify_l4_parent_inputs(workspace, identity.parent)
    return VerifiedL4Inputs(observed_runner, parent)
```

Keep `l4_contract.py` standard-library-only. `_read_simple_calibration_yaml` must accept exactly one non-empty scalar `val:` entry, reject duplicates/anchors/tags/collections, and return `{"val": value}`; the generated calibration file uses this restricted form. Wrap malformed file/schema failures from `OSError`, `KeyError`, `TypeError`, `ValueError`, and `json.JSONDecodeError` at the public boundary as `L4ContractError("L4 parent evidence is missing or malformed")`. `_contained_path` must resolve the candidate, require `candidate.relative_to(workspace)`, require `is_file()`, and translate a `ValueError` into `L4ContractError("<label> path escapes parent workspace")`.

Add `test_verify_l4_parent_inputs_imports_and_passes_without_third_party_modules`: launch the repository's system Python with `-I`, add only `src` to `sys.path`, block imports outside the standard library, and call `verify_l4_parent_inputs`. This is the pre-install contract used by Colab.

- [ ] **Step 9: Add parameterized mutation tests for every immutable dependency**

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "dirty_runner",
        "gate_bytes",
        "checkpoint_bytes",
        "onnx_bytes",
        "manifest_partition",
        "final_test_in_calibration",
        "calibration_image_bytes",
    ],
)
def test_verify_l4_inputs_rejects_mutation(
    tmp_path: Path, mutation: str
) -> None:
    repo, workspace, identity = _valid_contract_fixture(tmp_path)
    _apply_mutation(repo, workspace, mutation)
    with pytest.raises(L4ContractError):
        verify_l4_inputs(repo, workspace, identity)
```

Add a separate checkpoint-escape test that writes the malicious gate first, calculates that gate's exact SHA into a new `L4RunIdentity`, and then expects `L4ContractError("checkpoint path escapes parent workspace")`. This ensures the raw-gate check passes and path containment itself is exercised.

- [ ] **Step 10: Run the complete contract tests and Ruff checks**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_contract.py -q`

Expected: PASS.

Run: `uv run --locked --no-editable --extra train --group eval ruff check src/pcb_defect/l4_contract.py tests/test_l4_contract.py`

Expected: exit code 0.

- [ ] **Step 11: Commit Task 1**

```bash
git add src/pcb_defect/l4_contract.py tests/test_l4_contract.py
git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "feat: verify immutable L4 parent inputs"
```

### Task 2: Provenance-Bound L4 Benchmark

**Files:**
- Modify: `src/pcb_defect/benchmark.py:31-344`
- Modify: `tests/test_benchmark.py:1-52`

**Interfaces:**
- Consumes: `L4RunIdentity` and `verify_l4_inputs(repo: Path, workspace: Path, identity: L4RunIdentity) -> VerifiedL4Inputs` from Task 1.
- Produces: `benchmark(repo: Path, workspace: Path, identity: L4RunIdentity, *, warmup: int, cycles: int) -> Path`, `benchmark_is_complete(repo: Path, workspace: Path, identity: L4RunIdentity, report: dict[str, Any]) -> bool`, and a report at `benchmark_l4/<runner-prefix>/benchmark_l4.json`.
- Test-only helpers defined in `tests/test_benchmark.py`: `_fake_l4_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, L4RunIdentity, dict[str, object]]` installs deterministic fake contract/GPU/model/runtime/timer boundaries; `_configure_preflight_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> tuple[Path, Path, L4RunIdentity]`; `_completed_fake_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, L4RunIdentity, dict[str, Any]]`; and `_mutate_completed_benchmark(repo: Path, workspace: Path, report: dict[str, Any], mutation: str) -> None`.

- [ ] **Step 1: Extend the latency-summary test with the complete required statistics**

```python
def test_latency_summary_retains_complete_statistics() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "n_runs": 4,
        "mean_ms": 2.5,
        "std_ms": pytest.approx(1.2909944487358056),
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "min_ms": 1.0,
        "max_ms": 4.0,
        "fps_from_p50": 400.0,
    }
```

- [ ] **Step 2: Run the summary test and confirm the missing keys failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_benchmark.py::test_latency_summary_retains_complete_statistics -q`

Expected: FAIL because `std_ms`, `min_ms`, and `max_ms` are absent.

- [ ] **Step 3: Add sample standard deviation, minimum, and maximum to `summarize_latencies`**

```python
return {
    "n_runs": len(ordered),
    "mean_ms": statistics.fmean(ordered),
    "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    "p50_ms": p50,
    "p95_ms": _quantile(ordered, 0.95),
    "min_ms": ordered[0],
    "max_ms": ordered[-1],
    "fps_from_p50": 1000.0 / p50,
}
```

- [ ] **Step 4: Add a timing test proving 30 warmups, four complete cycles, synchronization, and raw retention**

```python
def test_time_backend_warms_then_measures_four_complete_cycles() -> None:
    calls: list[str] = []
    images = [object(), object()]
    result = _time_backend(
        lambda image: calls.append(f"infer:{images.index(image)}"),
        images,
        lambda: calls.append("sync"),
        warmup=30,
        cycles=4,
    )
    assert sum(item.startswith("infer:") for item in calls) == 38
    assert result["n_runs"] == 8
    assert len(result["raw_ms"]) == 8
    assert calls.count("sync") == 17
```

- [ ] **Step 5: Add CLI and report tests that require all five immutable expectations**

```python
def test_benchmark_report_records_runner_and_parent_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, identity, fake_runtime = _fake_l4_benchmark(tmp_path, monkeypatch)
    report_path = benchmark(repo, workspace, identity, warmup=30, cycles=4)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["runner_git_sha"] == identity.runner_git_sha
    assert report["experiment_git_sha"] == identity.parent.experiment_git_sha
    assert report["deployment_gate_sha256"] == identity.parent.deployment_gate_sha256
    assert report["artifacts"]["source_checkpoint_sha256"] == identity.parent.checkpoint_sha256
    assert report["artifacts"]["onnx_sha256"] == identity.parent.onnx_sha256
    assert report["runtime_contract"]["before"] == fake_runtime
    assert report["runtime_contract"]["after"] == fake_runtime
```

The fake boundary must monkeypatch `verify_l4_inputs`, `_load_gpu_runtime`, `_export_engine`, `_hardware`, `onnxruntime_state`, and `time.perf_counter`; it must not import Torch, Ultralytics, ONNX Runtime, CUDA, or TensorRT.

- [ ] **Step 6: Run the new provenance test and confirm the old function signature failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_benchmark.py::test_benchmark_report_records_runner_and_parent_provenance -q`

Expected: FAIL because `benchmark` does not accept `repo` and `identity`, and the report has only `git_sha`.

- [ ] **Step 7: Refactor the CLI and benchmark entry point to require immutable expectations**

```python
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--expected-runner-git-sha", required=True)
parser.add_argument("--expected-experiment-git-sha", required=True)
parser.add_argument("--expected-deployment-gate-sha256", required=True)
parser.add_argument("--expected-checkpoint-sha256", required=True)
parser.add_argument("--expected-onnx-sha256", required=True)
identity = L4RunIdentity.parse(
    runner_git_sha=args.expected_runner_git_sha,
    experiment_git_sha=args.expected_experiment_git_sha,
    deployment_gate_sha256=args.expected_deployment_gate_sha256,
    checkpoint_sha256=args.expected_checkpoint_sha256,
    onnx_sha256=args.expected_onnx_sha256,
)
benchmark(args.repo.resolve(), args.workspace.resolve(), identity, warmup=args.warmup, cycles=args.cycles)
```

Change the output directory to `workspace / "benchmark_l4" / identity.runner_git_sha[:12]`. Call `verify_l4_inputs` before creating it.

- [ ] **Step 8: Add fail-before-output tests for GPU, provider, TensorRT, and immutable-input failures**

```python
@pytest.mark.parametrize(
    "failure, message",
    [
        ("gpu", "benchmark requires a Colab L4"),
        ("cuda", "CUDA is unavailable"),
        ("provider", "CUDAExecutionProvider"),
        ("tensorrt", "TensorRT runtime is unavailable"),
        ("contract", "raw deployment-gate SHA-256 mismatch"),
    ],
)
def test_benchmark_preflight_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    repo, workspace, identity = _configure_preflight_failure(tmp_path, monkeypatch, failure)
    with pytest.raises((BenchmarkError, L4ContractError), match=message):
        benchmark(repo, workspace, identity, warmup=30, cycles=4)
    assert not (workspace / "benchmark_l4" / identity.runner_git_sha[:12]).exists()
```

- [ ] **Step 9: Implement the narrow GPU runtime boundary and preflight order**

```python
@dataclass(frozen=True, slots=True)
class _GpuRuntime:
    torch: Any
    yolo: Any
    onnx_model: Any
    tensorrt_version: str


def _load_gpu_runtime() -> _GpuRuntime:
    import tensorrt
    import torch
    from ultralytics import YOLO
    from pcb_defect.e2e_onnx import OnnxYoloModel

    return _GpuRuntime(torch, YOLO, OnnxYoloModel, tensorrt.__version__)
```

Perform preflight in this exact order before `output_dir.mkdir`:

```python
verified = verify_l4_inputs(repo, workspace, identity)
try:
    gpu = _load_gpu_runtime()
except ImportError as exc:
    raise BenchmarkError("TensorRT runtime is unavailable") from exc
if not gpu.torch.cuda.is_available():
    raise BenchmarkError("CUDA is unavailable")
device_name = gpu.torch.cuda.get_device_name(0)
if "L4" not in device_name:
    raise BenchmarkError(f"benchmark requires a Colab L4, found {device_name!r}")
runtime_before = onnxruntime_state(require_cuda_provider=True)
if not gpu.tensorrt_version:
    raise BenchmarkError("TensorRT runtime is unavailable")
output_dir.mkdir(parents=True)
```

- [ ] **Step 10: Make report publication and resume verification fail closed**

Write the report to an exclusive temporary path and publish only after all gates pass:

```python
runtime_after = onnxruntime_state(require_cuda_provider=True)
if runtime_after != runtime_before:
    raise BenchmarkError("ONNX Runtime state changed during the L4 benchmark")
fidelity_passed = abs(engine_delta) <= float(
    verified.parent.gate["fidelity"]["threshold"]
)
if not fidelity_passed:
    raise BenchmarkError("TensorRT FP16 calibration fidelity failed")
report = {
    "schema_version": "2.0",
    "status": "complete",
    "runner_git_sha": identity.runner_git_sha,
    "experiment_git_sha": identity.parent.experiment_git_sha,
    "deployment_gate_sha256": identity.parent.deployment_gate_sha256,
    "dataset_sha256": verified.parent.gate["dataset_sha256"],
    "manifest_sha256": verified.parent.gate["manifest_sha256"],
    "runtime_contract": {"before": runtime_before, "after": runtime_after},
    "protocol": protocol,
    "artifacts": artifacts,
    "fidelity": fidelity,
    "timings": timings,
    "environment": _environment(),
    "hardware": _hardware(gpu.torch, device_name, gpu.tensorrt_version),
}
temporary_report = report_path.with_name(".benchmark_l4.json.tmp")
with temporary_report.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary_report, report_path)
return report_path
```

`benchmark_is_complete` must rerun `verify_l4_inputs`, rehash the engine, compare both provenance fields and all expected hashes, require exact calibration hashes/counts, require unchanged runtime state, require `status == "complete"`, and return `False` for `BenchmarkError`, `L4ContractError`, file/schema errors, or any mismatch.

- [ ] **Step 11: Add resume mutation tests**

```python
@pytest.mark.parametrize(
    "mutation",
    ["runner_sha", "experiment_sha", "engine_bytes", "runtime_state", "calibration_bytes"],
)
def test_completed_benchmark_rejects_changed_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    repo, workspace, identity, report = _completed_fake_benchmark(tmp_path, monkeypatch)
    _mutate_completed_benchmark(repo, workspace, report, mutation)
    assert benchmark_is_complete(repo, workspace, identity, report) is False
```

- [ ] **Step 12: Run Task 2 tests and Ruff checks**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_benchmark.py -q`

Expected: PASS without GPU access.

Run: `uv run --locked --no-editable --extra train --group eval ruff check src/pcb_defect/benchmark.py tests/test_benchmark.py`

Expected: exit code 0.

- [ ] **Step 13: Commit Task 2**

```bash
git add src/pcb_defect/benchmark.py tests/test_benchmark.py
git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "feat: bind L4 benchmark to immutable provenance"
```

### Task 3: Atomic Private L4 Result Package

**Files:**
- Modify: `src/pcb_defect/result_package.py:22-70,130-161`
- Modify: `tests/test_result_package.py:13-44`
- Create: `src/pcb_defect/l4_package.py`
- Create: `tests/test_l4_package.py`

**Interfaces:**
- Consumes: `L4RunIdentity`, `verify_l4_inputs`, `benchmark_is_complete`, and `create_verifiable_zip`.
- Produces: `verify_verifiable_zip(package: Path) -> dict[str, Any]`, `l4_package_name(identity: L4RunIdentity) -> str`, `collect_l4_files(repo: Path, workspace: Path, identity: L4RunIdentity) -> list[Path]`, and `create_or_verify_l4_package(repo: Path, workspace: Path, output_root: Path, identity: L4RunIdentity) -> Path`.
- Test-only helpers defined in `tests/test_l4_package.py`: `_identity(*, runner: str, parent: str) -> L4RunIdentity`; `_complete_l4_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, L4RunIdentity]`, which creates every listed private file and stubs only the verified contract/benchmark boundaries; and `_corrupted_package(tmp_path: Path, corruption: str) -> Path`, which constructs the exact package/sidecar/archive corruption requested by the parameter.

- [ ] **Step 1: Write strict existing-package verification tests**

```python
def test_verify_verifiable_zip_rejects_missing_or_mutated_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    create_verifiable_zip(root, [Path("report.json")], package)
    sidecar = package.with_suffix(".zip.sha256")
    sidecar.write_text(f"{'0' * 64}  {package.name}\n", encoding="ascii")
    with pytest.raises(PackageError, match="sidecar bytes, name, or hash are invalid"):
        verify_verifiable_zip(package)
```

Use one parameterized corruption test for every archive boundary:

```python
@pytest.mark.parametrize(
    "corruption, message",
    [
        ("package_only", "must exist together"),
        ("sidecar_only", "must exist together"),
        ("manifest_json", "package manifest is invalid"),
        ("unlisted_member", "archive members do not match"),
        ("member_bytes", "archive member SHA-256 mismatch"),
    ],
)
def test_verify_verifiable_zip_rejects_corruption(
    tmp_path: Path, corruption: str, message: str
) -> None:
    package = _corrupted_package(tmp_path, corruption)
    with pytest.raises(PackageError, match=message):
        verify_verifiable_zip(package)
```

- [ ] **Step 2: Run the verifier test and confirm the missing function failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_result_package.py -q`

Expected: FAIL because `verify_verifiable_zip` is absent.

- [ ] **Step 3: Implement strict ZIP plus sidecar verification**

```python
def verify_verifiable_zip(package: Path) -> dict[str, Any]:
    package = package.resolve()
    sidecar = package.with_suffix(package.suffix + ".sha256")
    if package.exists() != sidecar.exists():
        raise PackageError("result package and SHA-256 sidecar must exist together")
    if not package.is_file():
        raise PackageError("result package is missing")
    package_sha256 = _sha256_file(package)
    expected_sidecar = f"{package_sha256}  {package.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise PackageError("result package sidecar bytes, name, or hash are invalid")
```

Open the ZIP, require exactly one `package_manifest.json`, require exact equality between listed and actual non-manifest members, and verify each member's SHA-256 and byte length before returning `{**manifest, "package_sha256": package_sha256}`.

- [ ] **Step 4: Add package naming and collection tests**

```python
def test_l4_package_name_contains_parent_and_runner_prefixes() -> None:
    identity = _identity(runner="a" * 40, parent="b" * 40)
    assert l4_package_name(identity) == (
        "paired-results-l4-bbbbbbbbbbbb-runner-aaaaaaaaaaaa.zip"
    )


def test_l4_collector_includes_only_private_verification_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    files = collect_l4_files(repo, workspace, identity)
    assert Path("benchmark_l4/aaaaaaaaaaaa/benchmark_l4.json") in files
    assert Path("benchmark_l4/aaaaaaaaaaaa/best_fp16.engine") in files
    assert Path("deployment/best.onnx") in files
    assert Path("runs/grouped/seed42/weights/best.pt") in files
    assert Path("inputs/paired_split_manifest.json") in files
    assert not any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in files)
```

- [ ] **Step 5: Run the L4 package tests and confirm the missing module failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_package.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'pcb_defect.l4_package'`.

- [ ] **Step 6: Implement the exact L4 private package collector**

```python
def l4_package_name(identity: L4RunIdentity) -> str:
    return (
        f"paired-results-l4-{identity.parent.experiment_git_sha[:12]}-"
        f"runner-{identity.runner_git_sha[:12]}.zip"
    )


def collect_l4_files(repo: Path, workspace: Path, identity: L4RunIdentity) -> list[Path]:
    verified = verify_l4_inputs(repo, workspace, identity)
    report_path = workspace / "benchmark_l4" / identity.runner_git_sha[:12] / "benchmark_l4.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not benchmark_is_complete(repo, workspace, identity, report):
        raise PackageError("L4 benchmark is incomplete or hash-mismatched")
    files = [
        Path("inputs/input_lock.json"),
        Path("inputs/paired_split_manifest.json"),
        Path("deployment/calibration.yaml"),
        Path("deployment/deployment_gate.json"),
        Path("deployment/model_contract.candidate.json"),
        verified.parent.onnx_path.relative_to(workspace),
        verified.parent.checkpoint_path.relative_to(workspace),
        report_path.relative_to(workspace),
        report_path.with_name("best_fp16.engine").relative_to(workspace),
        Path("l4_logs") / identity.runner_git_sha[:12] / "benchmark_command.log",
    ]
    for relative in files:
        resolved = (workspace / relative).resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError as exc:
            raise PackageError(f"L4 package path escapes workspace: {relative}") from exc
        if not resolved.is_file():
            raise PackageError(f"required L4 package file is missing: {relative}")
        if resolved.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise PackageError(f"dataset pixels are forbidden in the L4 package: {relative}")
    return files
```

Remove the optional `benchmark_l4` inclusion block from the legacy `_collect_result_files`; the A100 result package remains an A100 artifact, and all L4 evidence now goes through the provenance-aware `collect_l4_files`. Add a regression assertion to `test_collector_verifies_all_six_runs_and_includes_last_checkpoints` that no returned path starts with `benchmark_l4/`.

- [ ] **Step 7: Implement resumable create-or-verify behavior and CLI**

The CLI requires `--repo`, `--workspace`, `--output-root`, and the same five `--expected-*` identity flags as the benchmark. Derive the filename internally; never accept an arbitrary package name.

```python
def create_or_verify_l4_package(
    repo: Path, workspace: Path, output_root: Path, identity: L4RunIdentity
) -> Path:
    package = output_root.resolve() / l4_package_name(identity)
    sidecar = package.with_suffix(package.suffix + ".sha256")
    if package.exists() or sidecar.exists():
        verify_verifiable_zip(package)
        return package
    files = collect_l4_files(repo.resolve(), workspace.resolve(), identity)
    create_verifiable_zip(workspace.resolve(), files, package)
    verify_verifiable_zip(package)
    return package
```

- [ ] **Step 8: Add partial-pair, mutation, and no-overwrite tests**

```python
@pytest.mark.parametrize("existing", ["package", "sidecar"])
def test_l4_package_rejects_partial_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    repo, workspace, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    output.mkdir()
    package = output / l4_package_name(identity)
    target = package if existing == "package" else package.with_suffix(".zip.sha256")
    target.write_bytes(b"partial")
    with pytest.raises(PackageError, match="must exist together"):
        create_or_verify_l4_package(repo, workspace, output, identity)
```

- [ ] **Step 9: Run Task 3 tests and Ruff checks**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_result_package.py tests/test_l4_package.py -q`

Expected: PASS.

Run: `uv run --locked --no-editable --extra train --group eval ruff check src/pcb_defect/result_package.py src/pcb_defect/l4_package.py tests/test_result_package.py tests/test_l4_package.py`

Expected: exit code 0.

- [ ] **Step 10: Commit Task 3**

```bash
git add src/pcb_defect/result_package.py src/pcb_defect/l4_package.py tests/test_result_package.py tests/test_l4_package.py
git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "feat: package private L4 benchmark evidence"
```

### Task 4: Stage-Specific L4 Handoff and Thin Notebook

**Files:**
- Modify: `src/pcb_defect/handoff.py:105-225`
- Modify: `tests/test_handoff.py:71-229`
- Create: `src/pcb_defect/l4_handoff.py`
- Create: `tests/test_l4_handoff.py`
- Modify: `notebooks/deployment_benchmark_l4.ipynb`
- Modify: `tests/test_release_contract.py:159-194,362-417`

**Interfaces:**
- Consumes: `L4ParentIdentity`, `create_clean_bundle`, project handoff metadata, notebook rendering, `pcb_defect.benchmark`, `pcb_defect.l4_package`, runtime-contract helpers, and immutable parent constants.
- Produces: `create_l4_handoff(repo: Path, output_root: Path, parent: L4ParentIdentity) -> Path`, a directory named `colab-handoff-l4-<runner-prefix>`, one source bundle, one rendered notebook, and one manifest.
- Test-only helpers defined in `tests/test_l4_handoff.py`: `_parent_identity() -> L4ParentIdentity` returns the four approved parent values; `_complete_source_repo(tmp_path: Path) -> Path` creates a clean one-commit repository containing the real L4 notebook plus minimal protocol/base-model files; and `_minimal_handoff_repo(tmp_path: Path) -> Path` in `tests/test_handoff.py` creates the three current notebook templates for existing-CLI regression coverage.

- [ ] **Step 1: Write regression tests for reusable handoff helpers and unchanged existing CLI output**

Rename `_render_notebook` to `render_notebook` and extract `project_handoff_metadata(repo: Path) -> dict[str, Any]`; update the existing imports/tests. Add a regression test that calls the current `pcb_defect.handoff.main` and still receives the A100 and L4 notebooks, plus the probe notebook only when probe arguments are supplied.

```python
def test_existing_handoff_cli_preserves_paired_outputs(tmp_path: Path) -> None:
    repo = _minimal_handoff_repo(tmp_path)
    output = tmp_path / "paired-handoff"
    assert main(["--repo", str(repo), "--output", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {
        "handoff_manifest.json",
        "paired_experiment_a100.ipynb",
        "deployment_benchmark_l4.ipynb",
        "pcb-defect-source.bundle",
    }
```

- [ ] **Step 2: Run the handoff regression test and confirm it fails on the new public helper names**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_handoff.py -q`

Expected: FAIL until the imports and extracted helper are implemented.

- [ ] **Step 3: Extract the helpers without changing behavior**

```python
def project_handoff_metadata(repo: Path) -> dict[str, Any]:
    protocol = json.loads(
        (repo / "reports" / "protocol" / "paired_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    base_contract_path = repo / "configs" / "base_model.yaml"
    base_contract = yaml.safe_load(base_contract_path.read_text(encoding="utf-8"))
    return {
        "protocol_version": protocol["protocol_version"],
        "dataset_sha256": protocol["dataset"]["sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "base_model_contract_sha256": _sha256_file(base_contract_path),
        "base_model_source": base_contract["source"],
        "base_model_revision": base_contract["revision"],
        "base_model_sha256": base_contract["sha256"],
    }
```

Keep `_render_notebook = render_notebook` as a temporary compatibility alias only if an external import remains; repository tests and new code must use `render_notebook`.

- [ ] **Step 4: Write the stage-specific handoff failure and success tests**

```python
def test_l4_handoff_contains_only_l4_stage_files(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)
    output = create_l4_handoff(repo, tmp_path / "dist", _parent_identity())
    manifest = json.loads((output / "handoff_manifest.json").read_text(encoding="utf-8"))
    assert output.name == f"colab-handoff-l4-{manifest['snapshot_git_sha'][:12]}"
    assert {path.name for path in output.iterdir()} == {
        "deployment_benchmark_l4.ipynb",
        "handoff_manifest.json",
        "pcb-defect-source.bundle",
    }
    assert manifest["stage"] == "l4-benchmark"
    assert manifest["parent_experiment_git_sha"] == "9e3a1ed5827ac3759cbb15632f041e3e5c183b51"
    assert "a100_notebook" not in manifest
    assert "probe_notebook" not in manifest
```

Add tests that partial/malformed/mixed-stage arguments fail before `create_clean_bundle`, an existing final directory is not overwritten, source dirtiness fails, and a rendered notebook has no execution counts, outputs, or unresolved `PASTE_` sentinels.

- [ ] **Step 5: Run the L4 handoff tests and confirm the missing module failure**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_l4_handoff.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'pcb_defect.l4_handoff'`.

- [ ] **Step 6: Implement atomic stage-specific handoff generation**

```python
def create_l4_handoff(repo: Path, output_root: Path, parent: L4ParentIdentity) -> Path:
    repo = repo.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".l4-handoff-stage-", dir=output_root))
    content = staging / "content"
    metadata = {
        **project_handoff_metadata(repo),
        **{f"parent_{key}": value for key, value in parent.as_dict().items()},
        "stage": "l4-benchmark",
    }
    result = create_clean_bundle(repo, content, metadata)
    runner_sha = result["snapshot_git_sha"]
    final = output_root / f"colab-handoff-l4-{runner_sha[:12]}"
    if final.exists():
        raise HandoffError(f"refusing to overwrite handoff directory: {final}")
    drive_directory = (
        f"/content/drive/MyDrive/pcb-defect-paired/handoff-l4/{runner_sha[:12]}"
    )
    notebook_path = content / "deployment_benchmark_l4.ipynb"
    notebook_sha256 = render_notebook(
        repo / "notebooks" / "deployment_benchmark_l4.ipynb",
        notebook_path,
        {
            "PASTE_FINAL_BUNDLE_SHA256": result["bundle_sha256"],
            "PASTE_FINAL_GIT_SHA": runner_sha,
            "PASTE_PARENT_EXPERIMENT_GIT_SHA": parent.experiment_git_sha,
            "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": parent.deployment_gate_sha256,
            "PASTE_PARENT_CHECKPOINT_SHA256": parent.checkpoint_sha256,
            "PASTE_PARENT_ONNX_SHA256": parent.onnx_sha256,
            "PASTE_L4_HANDOFF_DIRECTORY": drive_directory,
        },
    )
    manifest_path = content / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "stage": "l4-benchmark",
            "l4_notebook": notebook_path.name,
            "l4_template_sha256": _sha256_file(
                repo / "notebooks" / "deployment_benchmark_l4.ipynb"
            ),
            "l4_notebook_sha256": notebook_sha256,
            "drive_handoff_directory": drive_directory,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _verify_l4_handoff(content, parent)
    content.replace(final)
    shutil.rmtree(staging)
    return final
```

Implement `_verify_l4_handoff(content: Path, parent: L4ParentIdentity) -> None` to run `git bundle verify`, clone into another function-owned temporary directory, assert one commit, exact detached runner SHA, clean tree, exact `.source_provenance.json`, exact manifest/file hashes, exactly three top-level handoff files, and no notebook output/execution count/sentinel. Wrap the body of `create_l4_handoff` in `try/finally`; on failure remove only `staging`, and on success remove the now-empty staging parent after `content.replace(final)`.

- [ ] **Step 7: Replace the L4 notebook's coupled constants with seven rendered values**

The first code cell must set hermetic controls before imports/actions and contain exactly these template assignments:

```python
SOURCE_BUNDLE_SHA256 = "PASTE_FINAL_BUNDLE_SHA256"
RUNNER_GIT_SHA = "PASTE_FINAL_GIT_SHA"
PARENT_EXPERIMENT_GIT_SHA = "PASTE_PARENT_EXPERIMENT_GIT_SHA"
PARENT_DEPLOYMENT_GATE_SHA256 = "PASTE_PARENT_DEPLOYMENT_GATE_SHA256"
PARENT_CHECKPOINT_SHA256 = "PASTE_PARENT_CHECKPOINT_SHA256"
PARENT_ONNX_SHA256 = "PASTE_PARENT_ONNX_SHA256"
L4_HANDOFF_DIRECTORY = "PASTE_L4_HANDOFF_DIRECTORY"
```

Derive `PARENT_WORKSPACE` only as:

```python
PARENT_WORKSPACE = (
    Path("/content/drive/MyDrive/pcb-defect-paired/workspaces")
    / PARENT_EXPERIMENT_GIT_SHA[:12]
)
```

- [ ] **Step 8: Write release-contract tests for notebook ordering and scope**

```python
def test_l4_notebook_separates_runner_and_parent_and_never_trains() -> None:
    code = _code(_notebook("notebooks/deployment_benchmark_l4.ipynb"))
    assert "RUNNER_GIT_SHA = \"PASTE_FINAL_GIT_SHA\"" in code
    assert "PARENT_EXPERIMENT_GIT_SHA = \"PASTE_PARENT_EXPERIMENT_GIT_SHA\"" in code
    assert "PARENT_WORKSPACE" in code
    assert "PARENT_WORKSPACE = DRIVE_ROOT / 'workspaces' / RUNNER_GIT_SHA[:12]" not in code
    assert "train-all" not in code
    assert "pcb_defect.experiment" not in code
    assert "YOLO_AUTOINSTALL'] = 'false'" in _first_code(
        _notebook("notebooks/deployment_benchmark_l4.ipynb")
    )
```

Use AST compilation for every code cell and assert no output or execution count exists.

- [ ] **Step 9: Implement the notebook's verify-install-benchmark-package flow**

The notebook must perform these actions in order:

1. Mount Drive; verify the bundle SHA before cloning.
2. Clone, detach checkout `RUNNER_GIT_SHA`, require clean status, and verify `HEAD`.
3. Before installing the project, add `REPO / "src"` to `sys.path`, construct `L4ParentIdentity`, and call the standard-library-only `verify_l4_parent_inputs(PARENT_WORKSPACE, parent)`; abort on any gate/checkpoint/ONNX/protocol/calibration mismatch.
4. Install `uv==0.11.18`; run locked, non-editable sync with `--extra train --group eval --reinstall-package pcb-defect`; use only `.venv/bin/python` afterward.
5. Construct `L4RunIdentity` and call `verify_l4_inputs(REPO, PARENT_WORKSPACE, identity)` before any benchmark output directory exists.
6. Record the strict runtime state, run `pcb_defect.benchmark` with all five expectations through `run_streaming_command`, recheck runtime equality, and verify `benchmark_is_complete`.
7. Run `pcb_defect.l4_package` with the same expectations, then call `verify_verifiable_zip` on the returned path.
8. Print `L4 HANDOFF COMPLETE` with the exact ZIP path and SHA-256 only after all verification returns successfully.

The benchmark command must be exactly assembled from immutable values:

```python
benchmark_command = [
    str(VENV_PYTHON), "-m", "pcb_defect.benchmark",
    "--repo", str(REPO),
    "--workspace", str(PARENT_WORKSPACE),
    "--expected-runner-git-sha", RUNNER_GIT_SHA,
    "--expected-experiment-git-sha", PARENT_EXPERIMENT_GIT_SHA,
    "--expected-deployment-gate-sha256", PARENT_DEPLOYMENT_GATE_SHA256,
    "--expected-checkpoint-sha256", PARENT_CHECKPOINT_SHA256,
    "--expected-onnx-sha256", PARENT_ONNX_SHA256,
    "--warmup", "30", "--cycles", "4",
]
```

Store the durable append-only benchmark log at `PARENT_WORKSPACE / "l4_logs" / RUNNER_GIT_SHA[:12] / "benchmark_command.log"`. Do not delete, rewrite, or repair any parent artifact or partial benchmark directory.

- [ ] **Step 10: Implement the L4 handoff CLI argument boundary**

The module CLI accepts only `--repo`, `--output-root`, and the four parent values. All four parent values are required; argparse rejects missing/extra stage flags before calling `create_l4_handoff`.

```python
parent = L4ParentIdentity.parse(
    experiment_git_sha=args.parent_experiment_git_sha,
    deployment_gate_sha256=args.parent_deployment_gate_sha256,
    checkpoint_sha256=args.parent_checkpoint_sha256,
    onnx_sha256=args.parent_onnx_sha256,
)
output = create_l4_handoff(args.repo, args.output_root, parent)
print(f"l4_handoff_dir={output}")
```

- [ ] **Step 11: Run Task 4 focused tests and render/compile checks**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_handoff.py tests/test_l4_handoff.py tests/test_release_contract.py -q`

Expected: PASS without GPU, dataset, weights, ONNX, TensorRT, API key, or network.

Run: `uv run --locked --no-editable --extra train --group eval ruff check src/pcb_defect/handoff.py src/pcb_defect/l4_handoff.py tests/test_handoff.py tests/test_l4_handoff.py tests/test_release_contract.py`

Expected: exit code 0.

- [ ] **Step 12: Commit Task 4**

```bash
git add src/pcb_defect/handoff.py src/pcb_defect/l4_handoff.py notebooks/deployment_benchmark_l4.ipynb tests/test_handoff.py tests/test_l4_handoff.py tests/test_release_contract.py
git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "feat: generate private L4 benchmark handoff"
```

### Task 5: Full CPU Verification and Immutable Handoff Generation

**Files:**
- Verify only: all tracked project files
- Generate ignored artifacts under: `dist/colab-handoff-l4-<runner-prefix>/`

**Interfaces:**
- Consumes: Tasks 1-4 at a clean committed HEAD.
- Produces: a locally verified, ignored L4 bundle/notebook/manifest directory ready for manual Google Drive upload. It does not produce public evidence or a GPU result.

- [ ] **Step 1: Refresh the locked non-editable environment**

Run: `uv sync --locked --no-editable --extra train --group eval --reinstall-package pcb-defect`

Expected: exit code 0 and no lockfile change.

- [ ] **Step 2: Run the complete CPU test suite**

Run: `uv run --locked --no-editable --extra train --group eval pytest -q`

Expected: every test passes; no test requires GPU, dataset, weights, ONNX, TensorRT, API key, or network.

- [ ] **Step 3: Run Ruff lint and format verification**

Run: `uv run --locked --no-editable --extra train --group eval ruff check .`

Expected: exit code 0.

Run: `uv run --locked --no-editable --extra train --group eval ruff format --check .`

Expected: exit code 0.

- [ ] **Step 4: Build wheel and source distribution into a unique temporary directory**

PowerShell:

```powershell
$l4BuildDirectory = Join-Path $env:TEMP ("pcb-defect-l4-build-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $l4BuildDirectory | Out-Null
uv build --out-dir $l4BuildDirectory
Get-ChildItem -LiteralPath $l4BuildDirectory
```

Expected: one wheel and one source archive are created outside the repository; tracked files remain unchanged.

- [ ] **Step 5: Run repository safety scans**

PowerShell:

```powershell
git diff --check
git status --short
git ls-files | rg "(?i)(^|/)(\.env|.*\.(pt|onnx|engine|plan|trt|zip|sha256))$"
git ls-files | rg "(?i)(^|/)(data|dataset|weights|exports|runs)/|\.(jpg|jpeg|png|bmp|tif|tiff)$"
rg -n "(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]" --glob "!.env" --glob "!uv.lock" .
rg -n "C:\\\\Users\\|/content/drive/MyDrive|/root/" README.md docs reports src tests notebooks scripts configs
git log --format="%H%x09%an%x09%ae%x09%cn%x09%ce%x09%(trailers:key=Co-authored-by,valueonly)"
```

Expected: clean status; no tracked model/export/result/dataset binary; no secret assignment; only intentional generic Colab paths inside notebook templates/specs/plans; every commit uses an approved identity and no unintended co-author.

- [ ] **Step 6: Generate the ignored immutable L4 handoff from clean HEAD**

Run:

```powershell
uv run --locked --no-editable --extra train --group eval python -m pcb_defect.l4_handoff `
  --repo . `
  --output-root dist `
  --parent-experiment-git-sha 9e3a1ed5827ac3759cbb15632f041e3e5c183b51 `
  --parent-deployment-gate-sha256 466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803 `
  --parent-checkpoint-sha256 44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a `
  --parent-onnx-sha256 b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8
```

Expected: exactly one new ignored `dist/colab-handoff-l4-<12 lowercase hex>/` directory containing `pcb-defect-source.bundle`, `deployment_benchmark_l4.ipynb`, and `handoff_manifest.json`.

- [ ] **Step 7: Independently inspect the generated handoff**

PowerShell:

```powershell
$l4Handoff = Get-ChildItem -LiteralPath dist -Directory -Filter "colab-handoff-l4-*" |
  Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
$l4Manifest = Get-Content -LiteralPath (Join-Path $l4Handoff.FullName "handoff_manifest.json") -Raw |
  ConvertFrom-Json
git bundle verify (Join-Path $l4Handoff.FullName "pcb-defect-source.bundle")
Get-FileHash -Algorithm SHA256 (Join-Path $l4Handoff.FullName "pcb-defect-source.bundle")
Get-FileHash -Algorithm SHA256 (Join-Path $l4Handoff.FullName "deployment_benchmark_l4.ipynb")
```

Expected: bundle and notebook hashes equal the manifest; parent identities equal the five approved values; runner and parent Git SHAs differ; stage is `l4-benchmark`; the notebook is unexecuted and contains no `PASTE_` sentinel.

- [ ] **Step 8: Confirm the final worktree is clean and no ignored artifact became tracked**

Run: `git status --short`

Expected: no output.

Run: `git ls-files dist`

Expected: no output.

- [ ] **Step 9: Record the handoff path and user instructions without committing generated files**

Report the exact local paths for the three files, instruct the user to upload them to the rendered Drive handoff directory, select a fresh Colab L4 runtime, and use **Run all**. State that the user should download both the final ZIP and `.sha256` only after `L4 HANDOFF COMPLETE` appears. Do not claim L4/TensorRT performance before the returned package passes a separate local audit.

## Final Acceptance Checklist

- [ ] Runner and parent identities are distinct in code, CLI, notebook, report, manifest, package name, and tests.
- [ ] The parent workspace is always derived from `9e3a1ed5827a`, never from the runner SHA.
- [ ] Gate, checkpoint, ONNX, dataset, protocol manifest, calibration list, and calibration image bytes fail closed on mutation.
- [ ] L4, CUDA provider, and TensorRT checks occur before benchmark output creation.
- [ ] PyTorch FP32, ONNX Runtime CUDA FP32, and TensorRT FP16 record a warmup count of 30 and retain raw timings for four 60-image cycles.
- [ ] Report statistics include count, mean, sample standard deviation, median, p95, min, and max.
- [ ] Fidelity uses only the calibration split and the deployment-gate threshold.
- [ ] Existing complete evidence is reusable only after all bindings verify; partial directories and partial package pairs fail.
- [ ] Generated ZIP and sidecar are private, deterministic, internally verified, and named with parent plus runner prefixes.
- [ ] Existing A100 and parity-probe tests remain green.
- [ ] Full pytest, Ruff lint, Ruff format, build, clean-tree, binary, secret, identity, and clean-bundle checks pass on CPU.
- [ ] No README, claims, model-card, release, remote, account, or hosted-service state changes in this phase.
