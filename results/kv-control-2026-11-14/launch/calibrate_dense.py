"""One fixed, untrained dense-code linear memory instrument calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

FROZEN_COMMIT = "ff62df0f85ff7cbf62f3209755ea066959aa7085"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codebook_sha256(keys, values):
    return hashlib.sha256(keys.detach().cpu().contiguous().numpy().tobytes()
                          + values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class DenseConfig:
    codebook_sha256: str
    adapter_source_sha256: str
    nkeys: int = 64
    nvalues: int = 16
    dim: int = 32
    seed: int = 90401
    step_size: float = 1.0

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def delta_update(keys, values, weight):
    error = values - torch.bmm(keys[:, None], weight)[:, 0]
    return weight + keys[:, :, None] * error[:, None, :]


class DenseDeltaAdapter(nn.Module):
    """Unit nonorthogonal Gaussian keys; exact step-one half-SSE updates."""
    method = "dense_delta"

    def __init__(self):
        super().__init__()
        from clmm.pretrain.kv_control import KVState
        self.state_type = KVState
        generator = torch.Generator(device="cpu").manual_seed(90401)
        keys = F.normalize(torch.randn(64, 32, generator=generator, dtype=torch.float32), dim=-1)
        values = torch.eye(32, dtype=torch.float32)[:16].clone()
        self.register_buffer("keys", keys)
        self.register_buffer("values", values)
        self.config = DenseConfig(codebook_sha256(keys, values), sha256(__file__))

    def _new(self, weight, count):
        return self.state_type({"weight": weight}, None, count, self.method, self.config.fingerprint)

    def _validate(self, state, *, on_device=True, finite=True):
        if (not isinstance(state, self.state_type) or state.method != self.method
                or state.config_fingerprint != self.config.fingerprint):
            raise ValueError("State method or configuration fingerprint mismatch")
        if (type(state.write_count) is not int or state.write_count < 0 or state.aux is not None
                or type(state.weights) is not dict or set(state.weights) != {"weight"}):
            raise ValueError("Invalid bounded dense state")
        weight = state.weights["weight"]
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 3 or weight.shape[0] < 1
                or tuple(weight.shape[1:]) != (32, 32) or weight.dtype != torch.float32
                or weight.layout != torch.strided):
            raise ValueError("State must contain float32 weights [batch, 32, 32]")
        if on_device and weight.device != self.keys.device:
            raise ValueError("State and model devices differ")
        if finite and not torch.isfinite(weight).all():
            raise ValueError("State weights must be finite")
        return weight

    def _ids(self, ids, size):
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
                or ids.device != self.keys.device or (ids < 0).any() or (ids >= size).any()):
            raise ValueError("IDs must be in-range int64 tensors on the model device")

    def _check_codebook(self):
        if codebook_sha256(self.keys, self.values) != self.config.codebook_sha256:
            raise ValueError("Frozen codebook fingerprint mismatch")

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("batch must be a positive integer")
        return self._new(self.keys.new_zeros(batch, 32, 32), 0)

    def write(self, keys, values, state):
        weight = self._validate(state)
        if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
                or keys.shape != (weight.shape[0],) or values.shape != keys.shape):
            raise ValueError("Write IDs must have shape [batch]")
        self._ids(keys, 64)
        self._ids(values, 16)
        return self._new(delta_update(self.keys[keys], self.values[values], weight), state.write_count + 1)

    def read(self, queries, state):
        weight = self._validate(state)
        if (not isinstance(queries, torch.Tensor) or queries.ndim != 2
                or queries.shape[0] != weight.shape[0] or queries.shape[1] < 1):
            raise ValueError("Queries must have shape [batch, queries]")
        self._ids(queries, 64)
        return torch.bmm(self.keys[queries], weight)[..., :16]

    def export_state(self, state):
        weight = self._validate(state)
        self._check_codebook()
        return dict(schema_version=1, method=self.method, config=self.config.to_dict(),
                    config_fingerprint=self.config.fingerprint, batch_size=weight.shape[0],
                    write_count=state.write_count, weights={"weight": weight.detach().cpu().clone()}, aux=None)

    def import_state(self, payload):
        fields = {"schema_version", "method", "config", "config_fingerprint", "batch_size",
                  "write_count", "weights", "aux"}
        if type(payload) is not dict or set(payload) != fields:
            raise ValueError("Invalid dense-state payload fields")
        expected_config = self.config.to_dict()
        valid_config = (type(payload["config"]) is dict and set(payload["config"]) == set(expected_config)
                        and all(type(payload["config"][key]) is type(value) and payload["config"][key] == value
                                for key, value in expected_config.items()))
        if (type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or type(payload["batch_size"]) is not int or payload["batch_size"] < 1
                or not valid_config):
            raise ValueError("Dense-state schema or configuration mismatch")
        self._check_codebook()
        state = self.state_type(payload["weights"], payload["aux"], payload["write_count"],
                                payload["method"], payload["config_fingerprint"])
        weight = self._validate(state, on_device=False)
        if weight.shape[0] != payload["batch_size"]:
            raise ValueError("Dense-state batch size mismatch")
        return self._new(weight.detach().to(self.keys.device).clone(), state.write_count)


def load_frozen(repo):
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != FROZEN_COMMIT:
        raise AssertionError("Calibration requires frozen repository HEAD ff62df0")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=repo).returncode != 0:
        raise AssertionError("Frozen repository has tracked changes")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    runner = importlib.import_module("scripts.kv_representation_control")
    streams = importlib.import_module("clmm.pretrain.kv_streams")
    for module, relative in ((runner, "scripts/kv_representation_control.py"),
                             (streams, "clmm/pretrain/kv_streams.py")):
        if Path(module.__file__).resolve() != repo / relative:
            raise AssertionError("Imported model source is outside the frozen repository")
    return runner, streams


def main(args):
    if not 0 < args.max_seconds <= 300:
        raise ValueError("max-seconds must be in (0, 300]")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    repo, out = args.repo.resolve(), args.out.resolve()
    if out == repo or repo in out.parents:
        raise ValueError("Calibration output must be outside the frozen repository")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    command=[sys.executable, *sys.argv], source_commit=FROZEN_COMMIT,
                    args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    method="dense_delta", offline_training=False, slow_learned_parameters=0,
                    representation="64 fixed normalized Gaussian keys in 32 dimensions; one-hot values in the first 16 coordinates",
                    interpretation="Untrained fixed linear instrument calibration; not a full Titans model.",
                    fast_state_bytes_per_stream=4096, logical_buffer_pairs=8, logical_snapshot_pairs=4)
    timer = hasattr(signal, "setitimer")
    previous_handler = None
    def expired(_signum, _frame):
        raise TimeoutError("Dense calibration exceeded its internal wall-clock limit")
    try:
        if timer:
            previous_handler = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.perf_counter()))
        runner, streams = load_frozen(repo)
        common = runner.common
        metadata["source_sha256"] = common.capture_sources(out)
        runner_name = "scripts/kv_representation_control.py"
        shutil.copy2(repo / runner_name, out / "sources" / runner_name)
        metadata["source_sha256"][runner_name] = sha256(repo / runner_name)
        shutil.copy2(__file__, out / "calibrate_dense.py")
        metadata["adapter_source_sha256"] = sha256(__file__)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(90401)
        device = torch.device(args.device)
        model = DenseDeltaAdapter().to(device)
        metadata["config"] = model.config.to_dict()
        metadata["environment"] = dict(python=sys.version, torch=str(torch.__version__),
            numpy=runner.np.__version__, device=str(device), deterministic=True, tf32=False,
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
        _, _, seen = streams.training_data(1200, 32, 90302)
        common.json_file(out / "planned_training_exclusion.json", dict(train_seed=90302, planned_steps=1200,
            batch_size=32, used_for_offline_training=False, signatures=sorted(seen)))
        metadata["planned_training_exclusion"] = dict(train_seed=90302, steps=1200, batch_size=32,
            unique_signatures=len(seen), sha256=sha256(out / "planned_training_exclusion.json"))
        torch.save(dict(format="clmm.kv.dense-calibration.v1", method=model.method,
            config=model.config.to_dict(), model_state={key: value.cpu() for key, value in model.state_dict().items()},
            source_commit=FROZEN_COMMIT, adapter_source_sha256=sha256(__file__), offline_training=False), out / "checkpoint.pt")
        metadata["checkpoint_sha256"] = sha256(out / "checkpoint.pt")
        evaluation_args = argparse.Namespace(dev_seed=90403, eval_episodes=128,
            pressure_episodes=32, eval_batch_size=16, suite="full")
        metadata["evaluation_args"] = vars(evaluation_args)
        common.json_file(out / "metadata.json", metadata)
        metadata["evaluation"] = runner.evaluate(model, evaluation_args, out / "evaluation", seen, deadline)
        for name, digest in metadata["source_sha256"].items():
            if sha256(repo / name) != digest or sha256(out / "sources" / name) != digest:
                raise AssertionError("Frozen evaluator source changed")
        if (sha256(__file__) != metadata["adapter_source_sha256"]
                or sha256(out / "calibrate_dense.py") != metadata["adapter_source_sha256"]):
            raise AssertionError("Dense adapter source changed")
        common.before_deadline(deadline)
        metadata["status"] = "succeeded"
    except BaseException as error:
        metadata.update(status="failed", failure=repr(error), traceback=traceback.format_exc())
    finally:
        if timer and previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        metadata.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return 0 if metadata["status"] == "succeeded" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--max-seconds", type=float, default=300)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
