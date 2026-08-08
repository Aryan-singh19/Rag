import os
import numpy as np
import torch

from dataclasses import asdict
from pathlib import Path

RESUME_FILENAME = "resume_latest.pt"


def optimizer_state(agent) -> dict:
    '''
    Return optimizer state. PlaNet has one optimizer; Dreamer stores the model,
    actor, and value optimizer states by name.
    '''
    if hasattr(agent, "named_optimizers"):
        return {name: opt.state_dict() for name, opt in agent.named_optimizers().items()}
    return agent.optimizer.state_dict()


def load_optimizer_state(agent, state: dict) -> None:
    '''
    Load optimizer state. A missing Dreamer optimizer entry raises KeyError
    because the checkpoint does not match the agent.
    '''
    if hasattr(agent, "named_optimizers"):
        for name, opt in agent.named_optimizers().items():
            opt.load_state_dict(state[name])
    else:
        agent.optimizer.load_state_dict(state)


def atomic_torch_save(payload: dict, path: Path) -> None:
    '''
    Save to a temporary file, then replace the target. If saving is
    interrupted, the previous checkpoint remains intact.
    '''
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def collect_rng_states(env, eval_env) -> dict:
    '''
    Save the random-number states used by NumPy, PyTorch, and each DMC
    environment. Skip environment random states that are not exposed.
    '''
    states = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        states["torch_mps"] = torch.mps.get_rng_state()
    for name, e in (("env", env), ("eval_env", eval_env)):
        task_random = getattr(getattr(e, "task", None), "random", None)
        if task_random is not None:
            states[f"{name}_task"] = task_random.get_state()
    return states


def restore_rng_states(states: dict, env, eval_env) -> None:
    '''
    Restore saved random-number states that are available on this machine.
    For example, skip CUDA state when loading on CPU.
    '''
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    if "torch_mps" in states and torch.backends.mps.is_available():
        torch.mps.set_rng_state(states["torch_mps"])
    for name, e in (("env", env), ("eval_env", eval_env)):
        key = f"{name}_task"
        task_random = getattr(getattr(e, "task", None), "random", None)
        if key in states and task_random is not None:
            task_random.set_state(states[key])


def save_resume_checkpoint(
    checkpoint_dir: Path,
    cfg,
    agent,
    replay,
    env,
    eval_env,
    step: int,
    episodes_collected: int,
    transitions_collected: int,
    env_steps_collected: int,
    best_eval_return: float,
    wandb_run_id: str | None,
) -> Path:
    '''
    Save everything required to resume in one file.

    Keeping the model, optimizer, replay, and random-number state together
    prevents mixing different training steps. Replay retains pixels because
    MuJoCo state alone cannot restore every randomized visual detail. Each
    save replaces the previous resume checkpoint.
    '''
    payload = {
        "config": asdict(cfg),
        "agent_type": getattr(agent, "agent_type", "planet"),
        "model": agent.state_dict(),
        "optimizer": optimizer_state(agent),
        "replay": replay.state_dict(),
        "step": step,
        "episodes_collected": episodes_collected,
        "transitions_collected": transitions_collected,
        "env_steps_collected": env_steps_collected,
        "best_eval_return": best_eval_return,
        "wandb_run_id": wandb_run_id,
        "rng": collect_rng_states(env, eval_env),
    }
    path = Path(checkpoint_dir) / RESUME_FILENAME
    atomic_torch_save(payload, path)
    return path


def load_resume_checkpoint(path: Path, agent, replay, env, eval_env) -> dict:
    '''
    Restore the agent, optimizer, replay buffer, and random-number states.
    Return the saved data so training can restore counters and the W&B run
    ID.

    weights_only=False is required because the checkpoint contains NumPy RNG
    state and other Python objects. Load only checkpoint files you trust.
    '''
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Load replay first so invalid replay data leaves the agent unchanged.
    replay.load_state_dict(payload["replay"])
    agent.load_state_dict(payload["model"])
    load_optimizer_state(agent, payload["optimizer"])
    restore_rng_states(payload["rng"], env, eval_env)
    return payload


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    '''
    Return resume_latest.pt from the checkpoint directory if it exists.
    '''
    path = Path(checkpoint_dir) / RESUME_FILENAME
    return path if path.exists() else None


def milestones_crossed(
    milestone_env_steps: tuple[int, ...],
    prev_env_steps: int,
    env_steps: int,
) -> list[int]:
    '''
    Return the environment-step milestones crossed during the latest
    collection. Steps increase by full episodes, so check an interval instead
    of exact equality.
    '''
    return [m for m in milestone_env_steps if prev_env_steps < m <= env_steps]


def save_milestone(
    checkpoint_dir: Path,
    cfg,
    agent,
    step: int,
    env_steps: int,
    target: int,
) -> Path:
    '''
    Save a model-only checkpoint at an environment-step milestone. It omits
    optimizer and replay data to keep the file small.
    '''
    payload = {
        "config": asdict(cfg),
        "agent_type": getattr(agent, "agent_type", "planet"),
        "model": agent.state_dict(),
        "step": step,
        "env_steps": env_steps,
        "milestone_env_steps": target,
    }
    path = Path(checkpoint_dir) / f"milestone_env{target}.pt"
    atomic_torch_save(payload, path)
    return path
