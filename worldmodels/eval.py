'''
Evaluate a saved checkpoint by rebuilding the agent, running episodes, and
reporting returns.

Usage:
    python -m worldmodels.eval --checkpoint checkpoints/planet/cartpole_swingup/seed_0/best.pt
    python -m worldmodels.eval --checkpoint <path> --episodes 20 --seed 7 --json

Supported files are best.pt, final.pt, milestone_env*.pt, and
resume_latest.pt.

--override changes evaluation settings, for example the CEM horizon or number
of candidates. Results record all overrides so runs with different settings
remain distinguishable.
'''
import argparse
import json
import sys

import numpy as np
import torch

from dataclasses import fields, replace
from pathlib import Path

from worldmodels.agents import PlaNet
from worldmodels.config import PlaNetTrainConfig, DreamerTrainConfig
from worldmodels import trainer


def config_class_for(payload: dict):
    '''
    Return the config class selected by the checkpoint's agent_type field.
    If the field is absent, use PlaNet.
    '''
    agent_type = payload.get("agent_type", "planet")
    if agent_type == "planet":
        return PlaNetTrainConfig
    if agent_type == "dreamer":
        return DreamerTrainConfig
    raise ValueError(f"unknown agent_type in checkpoint: {agent_type!r}")


def config_from_payload(payload: dict) -> tuple[PlaNetTrainConfig | DreamerTrainConfig, bool]:
    '''
    Rebuild the training config stored in a checkpoint.

    Return (config, has_config). If the checkpoint has no config, use the
    defaults for its agent type and return False.
    '''
    config_cls = config_class_for(payload)
    stored = payload.get("config")
    if stored is None:
        return config_cls(), False

    field_names = {f.name for f in fields(config_cls)}
    kwargs = {k: v for k, v in stored.items() if k in field_names}
    # Accept list-valued checkpoint fields where the config expects tuples.
    if "wandb_tags" in kwargs:
        kwargs["wandb_tags"] = tuple(kwargs["wandb_tags"])
    if "milestone_env_steps" in kwargs:
        kwargs["milestone_env_steps"] = tuple(kwargs["milestone_env_steps"])
    return config_cls(**kwargs), True


def apply_overrides(cfg: PlaNetTrainConfig, overrides: dict) -> PlaNetTrainConfig:
    '''
    apply eval-time config overrides with field-name and type validation

    Values arrive as strings from the CLI or as JSON scalars from run
    configs; both are coerced through the dataclass field's declared type.
    '''
    if not overrides:
        return cfg

    field_types = {f.name: f.type for f in fields(type(cfg))}
    coerced = {}
    for name, value in overrides.items():
        if name not in field_types:
            raise ValueError(f"unknown config field in override: {name}")
        if isinstance(value, bool):
            raise ValueError(f"boolean value for override {name!r}; no config field takes a bool")
        current = getattr(cfg, name)
        if isinstance(current, bool):
            raise ValueError(f"boolean overrides not supported: {name}")
        try:
            if isinstance(current, int):
                # never truncate: 10.5 into an int field is a config error,
                # not a rounding opportunity
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError(f"override {name!r} must be an integer, got {value}")
                coerced[name] = int(value)
            elif isinstance(current, float):
                coerced[name] = float(value)
            elif isinstance(current, str):
                if not isinstance(value, str):
                    raise ValueError(f"override {name!r} must be a string, got {value!r}")
                coerced[name] = value
            else:
                raise ValueError(f"field {name} is not overridable at eval time")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(("override", "field")):
                raise
            raise ValueError(f"override {name!r}: cannot coerce {value!r} ({exc})")
    return replace(cfg, **coerced)


def build_agent(cfg, action_dim: int, device, agent_type: str = "planet"):
    '''
    Build the agent selected by agent_type.
    '''
    if agent_type == "dreamer":
        from worldmodels.trainer.dreamer_trainer import build_dreamer
        return build_dreamer(cfg, action_dim, device)
    return PlaNet(
        s_dim=cfg.s_dim,
        h_dim=cfg.h_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mlp_layers=cfg.mlp_layers,
        mlp_activation=cfg.mlp_activation,
        action_dim=action_dim,
        device=device,
        planning_horizon=cfg.planning_horizon,
        num_candidates=cfg.num_candidates,
        num_elites=cfg.num_elites,
        refine_iterations=cfg.refine_iterations,
        cem_min_std=cfg.cem_min_std,
        encoded_dim=cfg.encoded_dim,
        min_std=cfg.min_std,
        learning_rate=cfg.learning_rate,
        adam_epsilon=cfg.adam_epsilon,
        gradient_clip_norm=cfg.gradient_clip_norm,
        free_nats=cfg.free_nats,
        kl_loss_scale=cfg.kl_loss_scale,
        reconstruction_loss_scale=cfg.reconstruction_loss_scale,
        reward_loss_scale=cfg.reward_loss_scale,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int | None = None,
    seed: int | None = None,
    overrides: dict | None = None,
) -> dict:
    '''
    Load a checkpoint and run evaluation episodes.

    Return per-episode returns, summary statistics, checkpoint and config
    details, the evaluation seed, and applied overrides.
    '''
    from dm_control import suite
    from worldmodels.trainer import checkpointing
    from worldmodels.trainer.planet_trainer import evaluate_returns

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"{checkpoint_path} is missing the required 'model' entry")

    cfg, has_config = config_from_payload(payload)
    cfg = apply_overrides(cfg, overrides or {})
    if not has_config:
        print(
            f"WARNING: checkpoint has no training config; using "
            f"{type(cfg).__name__} defaults. Results are valid only if those "
            "defaults match the training run.",
            file=sys.stderr,
        )

    if episodes is None:
        episodes = payload.get("eval_episodes", cfg.eval_episodes)
    if seed is None:
        seed = payload.get("eval_seed", cfg.seed)

    saved_rng_state = checkpointing.collect_rng_states(None, None)
    try:
        device = trainer.select_device()

        with suite.load(
            domain_name=cfg.domain,
            task_name=cfg.task,
            task_kwargs={"random": seed},
        ) as env:
            action_dim = env.action_spec().shape[0]
            agent = build_agent(cfg, action_dim, device, payload.get("agent_type", "planet"))
            agent.load_state_dict(payload["model"])
            returns = evaluate_returns(
                cfg,
                env,
                agent,
                episodes=episodes,
                seed=seed,
            )
            for i, episode_return in enumerate(returns):
                print(f"episode {i + 1}/{episodes}: return={episode_return:.1f}", file=sys.stderr)
    finally:
        checkpointing.restore_rng_states(saved_rng_state, None, None)

    return {
        "checkpoint": str(checkpoint_path),
        "agent_type": payload.get("agent_type", "planet"),
        "domain": cfg.domain,
        "task": cfg.task,
        "checkpoint_contains_config": has_config,
        "overrides": dict(overrides or {}),
        "checkpoint_step": payload.get("step"),
        "checkpoint_env_steps": payload.get(
            "env_steps", payload.get("env_steps_collected")
        ),
        "eval_seed": seed,
        "episodes": episodes,
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="evaluation episodes (default: config's eval_episodes)")
    parser.add_argument("--seed", type=int, default=None, help="evaluation seed (default: checkpoint's recorded seed)")
    parser.add_argument("--json", action="store_true", help="print the result dict as one JSON line on stdout")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="eval-time config override, repeatable (e.g. --override planning_horizon=6)",
    )
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        if "=" not in item:
            parser.error(f"--override expects FIELD=VALUE, got: {item}")
        name, value = item.split("=", 1)
        overrides[name] = value

    result = evaluate_checkpoint(Path(args.checkpoint), args.episodes, args.seed, overrides)

    if args.json:
        print(json.dumps(result))
    else:
        for key in ("checkpoint", "domain", "task", "checkpoint_step", "checkpoint_env_steps",
                    "eval_seed", "episodes", "overrides",
                    "return_mean", "return_std", "return_min", "return_max"):
            print(f"{key}={result[key]}")
