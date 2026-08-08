import numpy as np
import torch
import argparse
import wandb

from dataclasses import asdict, replace
from tqdm import tqdm
from pathlib import Path

from worldmodels.agents import Dreamer
from worldmodels.utils import ReplayBuffer
from worldmodels.config import DreamerTrainConfig, DreamerDebugConfig
from worldmodels import trainer
from worldmodels.trainer import checkpointing
from worldmodels.trainer.planet_trainer import (
    get_planet_episode,
    compute_metrics,
    eval_agent,
    write_planet_imagination_video,
)


def build_dreamer(cfg: DreamerTrainConfig | DreamerDebugConfig, action_dim: int, device) -> Dreamer:
    '''
    Build a Dreamer agent from a training configuration.
    '''
    return Dreamer(
        s_dim=cfg.s_dim,
        h_dim=cfg.h_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mlp_layers=cfg.mlp_layers,
        mlp_activation=cfg.mlp_activation,
        action_dim=action_dim,
        device=device,
        encoded_dim=cfg.encoded_dim,
        min_std=cfg.min_std,
        free_nats=cfg.free_nats,
        kl_loss_scale=cfg.kl_loss_scale,
        reconstruction_loss_scale=cfg.reconstruction_loss_scale,
        reward_loss_scale=cfg.reward_loss_scale,
        reward_hidden_dim=cfg.reward_hidden_dim,
        reward_layers=cfg.reward_layers,
        value_hidden_dim=cfg.value_hidden_dim,
        value_layers=cfg.value_layers,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_layers=cfg.actor_layers,
        actor_init_std=cfg.actor_init_std,
        actor_mean_scale=cfg.actor_mean_scale,
        actor_min_std=cfg.actor_min_std,
        imagination_horizon=cfg.imagination_horizon,
        discount_gamma=cfg.discount_gamma,
        return_lambda=cfg.return_lambda,
        model_lr=cfg.model_lr,
        actor_lr=cfg.actor_lr,
        value_lr=cfg.value_lr,
        adam_epsilon=cfg.adam_epsilon,
        gradient_clip_norm=cfg.gradient_clip_norm,
    )


def train_dreamer(cfg: DreamerTrainConfig | DreamerDebugConfig, resume: str | None = None):
    from dm_control import suite

    '''
    Train Dreamer by updating its world model, actor, and value model from
    replay and imagined trajectories.
    '''
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = trainer.select_device()
    print(f"Using device: {device}\n")

    with (
            suite.load(domain_name=cfg.domain,
            task_name=cfg.task,
            task_kwargs={
                "random": cfg.seed
            }
        ) as env,
            suite.load(domain_name=cfg.domain,
            task_name=cfg.task,
            task_kwargs={
                "random": cfg.seed
            }
        ) as eval_env):

        action_spec = env.action_spec()
        action_dim = action_spec.shape[0]

        agent = build_dreamer(cfg, action_dim, device)

        replay = ReplayBuffer(
            capacity=cfg.replay_capacity,
            image_bits=cfg.image_bits,
        )

        checkpoint_dir = Path(cfg.checkpoint_dir) / f"{cfg.domain}_{cfg.task}" / f"seed_{cfg.seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        resume_payload = None
        if resume:
            resume_path = Path(resume)
            if resume_path.is_dir():
                resume_path = resume_path / checkpointing.RESUME_FILENAME
            resume_payload = checkpointing.load_resume_checkpoint(
                resume_path, agent, replay, env, eval_env
            )
            print(
                f"Resumed from {resume_path}: "
                f"step={resume_payload['step']}, "
                f"env_steps={resume_payload['env_steps_collected']}, "
                f"replay_transitions={replay.num_steps}\n"
            )

        with wandb.init(
            entity=cfg.entity,
            project=cfg.project,
            name=cfg.run_name,
            tags=cfg.wandb_tags,
            config=asdict(cfg),
            mode=cfg.wandb_mode,
            id=resume_payload["wandb_run_id"] if resume_payload else None,
            resume="must" if resume_payload and resume_payload["wandb_run_id"] else None,
        ) as run:

            model_path = checkpoint_dir / "model.txt"
            model_path.write_text(str(agent) + "\n", encoding="utf-8")
            run.save(str(model_path), base_path=str(checkpoint_dir))

            if resume_payload is None:
                start_step = 0
                episodes_collected = 0
                transitions_collected = 0
                env_steps_collected = 0
                best_eval_return = float("-inf")
            else:
                start_step = resume_payload["step"]
                episodes_collected = resume_payload["episodes_collected"]
                transitions_collected = resume_payload["transitions_collected"]
                env_steps_collected = resume_payload["env_steps_collected"]
                best_eval_return = resume_payload["best_eval_return"]
            loss_hist = []

            wandb_run_id = getattr(run, "id", None)

            if resume_payload is None:
                for _ in range(cfg.seed_episodes):
                    episode = trainer.collect_random_episode(env, cfg)
                    replay.add_episode(*episode)

                    num_transitions = episode[1].shape[0]
                    episodes_collected += 1
                    transitions_collected += num_transitions
                    env_steps_collected += num_transitions * cfg.action_repeat

            for step in tqdm(
                range(start_step + 1, cfg.train_steps + 1),
                initial=start_step,
                total=cfg.train_steps,
            ):
                obs, actions, rewards = replay.sample_batch(
                    batch_size=cfg.batch_size,
                    seq_len=cfg.sequence_length,
                    device=agent.device,
                )

                losses = agent.update(obs, actions, rewards)
                loss_hist.append(losses)

                if step % cfg.collect_interval == 0:
                    prev_env_steps = env_steps_collected
                    for _ in range(cfg.episodes_per_collection):
                        episode = get_planet_episode(env, agent, cfg, training=True)
                        replay.add_episode(*episode)
                        num_transitions = episode[1].shape[0]
                        episodes_collected += 1
                        transitions_collected += num_transitions
                        env_steps_collected += num_transitions * cfg.action_repeat

                    for target in checkpointing.milestones_crossed(
                        cfg.milestone_env_steps, prev_env_steps, env_steps_collected
                    ):
                        milestone_path = checkpointing.save_milestone(
                            checkpoint_dir, cfg, agent, step,
                            env_steps_collected, target,
                        )
                        tqdm.write(f"saved milestone checkpoint: {milestone_path}")

                if step % cfg.log_interval == 0:
                    metric, formatted = compute_metrics(
                        loss_hist,
                        replay.num_steps,
                        episodes_collected,
                        transitions_collected,
                        env_steps_collected
                    )
                    loss_hist.clear()
                    run.log(metric, step=step)
                    tqdm.write("--------------------logging metrics--------------------")
                    tqdm.write(f"step={step}\n{formatted}")

                if step % cfg.eval_interval == 0:
                    tqdm.write("-------------evaluating policy performance-------------")
                    with suite.load(
                        domain_name=cfg.domain,
                        task_name=cfg.task,
                        task_kwargs={"random": cfg.seed},
                    ) as score_env:
                        eval_metric, formatted = eval_agent(cfg, score_env, agent)
                    eval_metric["eval/env_steps"] = env_steps_collected
                    formatted += f"\neval/env_steps={env_steps_collected}"
                    run.log(eval_metric, step=step)
                    tqdm.write(f"step={step}\n{formatted}")

                    if eval_metric["eval/return_mean"] > best_eval_return:
                        best_eval_return = eval_metric["eval/return_mean"]
                        checkpointing.atomic_torch_save({
                            "config": asdict(cfg),
                            "agent_type": agent.agent_type,
                            "model": agent.state_dict(),
                            "step": step,
                            "env_steps": env_steps_collected,
                            "eval_seed": cfg.seed,
                            "eval_episodes": cfg.eval_episodes,
                            "eval_return": best_eval_return,
                        }, checkpoint_dir / "best.pt")

                    video_path = write_planet_imagination_video(cfg, eval_env, agent, step)
                    run.log(
                        {"video/actual_vs_imagined": wandb.Video(
                            str(video_path), format='mp4'
                        )},
                        step=step
                    )

                if step % cfg.checkpoint_interval == 0:
                    checkpointing.save_resume_checkpoint(
                        checkpoint_dir, cfg, agent, replay, env, eval_env,
                        step=step,
                        episodes_collected=episodes_collected,
                        transitions_collected=transitions_collected,
                        env_steps_collected=env_steps_collected,
                        best_eval_return=best_eval_return,
                        wandb_run_id=wandb_run_id,
                    )

            checkpointing.atomic_torch_save(
                {
                    "config": asdict(cfg),
                    "agent_type": agent.agent_type,
                    "model": agent.state_dict(),
                    "step": cfg.train_steps,
                    "env_steps": env_steps_collected,
                },
                checkpoint_dir / "final.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run the small debug configuration",
    )
    parser.add_argument("--domain", default=None, help="DMC domain override")
    parser.add_argument("--task", default=None, help="DMC task override")
    parser.add_argument("--seed", type=int, default=None, help="training seed override")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "resume from a checkpoint: a resume_latest.pt path, a run's "
            "checkpoint dir, or 'auto' to look inside this config's own "
            "checkpoint dir"
        ),
    )
    args = parser.parse_args()

    cfg = DreamerDebugConfig() if args.debug else DreamerTrainConfig()
    cfg = replace(
        cfg,
        domain=args.domain if args.domain is not None else cfg.domain,
        task=args.task if args.task is not None else cfg.task,
        seed=args.seed if args.seed is not None else cfg.seed,
    )

    if args.wandb_mode is not None:
        cfg = replace(cfg, wandb_mode=args.wandb_mode)

    resume = args.resume
    if resume == "auto":
        run_dir = Path(cfg.checkpoint_dir) / f"{cfg.domain}_{cfg.task}" / f"seed_{cfg.seed}"
        found = checkpointing.find_resume_checkpoint(run_dir)
        if found is None:
            print(f"--resume auto: no resume checkpoint in {run_dir}, starting fresh\n")
            resume = None
        else:
            resume = str(found)

    train_dreamer(cfg, resume=resume)
