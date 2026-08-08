import numpy as np
import torch
import torch.nn as nn
import einops as op
import argparse
import wandb

from dataclasses import asdict, replace
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

from dm_env import Environment, TimeStep, specs

from worldmodels.agents import PlaNet
from worldmodels.models import RSSMState
from worldmodels.utils import ReplayBuffer
from worldmodels.config import PlaNetTrainConfig, PlaNetDebugConfig
from worldmodels import trainer
from worldmodels.trainer import checkpointing

def get_planet_episode(env, agent:nn.Module, cfg:PlaNetTrainConfig, training:bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Collect one episode using a PlaNet or Dreamer agent.

    Inputs:
    env: a DeepMind Control Suite environment
    agent: PlaNet or Dreamer agent
    cfg: training config

    Returns:
    obs (T+1, H, W, 3): raw observation images
    actions (T, action_dim): actions for transitions obs[:-1] -> obs[1:]
    rewards (T): rewards emitted AFTER each action
    physics (T+1, state_dim): MuJoCo state per frame, aligned with obs
    '''
    time_step = env.reset()
    action_spec:specs.BoundedArray = env.action_spec()
    action_dim = action_spec.shape[0]

    obs = []
    actions = []
    rewards = []
    physics = []

    observation = trainer.render_pixels(env, cfg.image_size, cfg.camera_id)
    obs.append(observation)
    physics.append(trainer.get_physics_state(env))
    prev_action:torch.Tensor = torch.zeros((1, action_dim), device=agent.device)
    prev_state = agent.rssm.initialize_RSSM_state(1)

    max_decisions = cfg.episode_length // cfg.action_repeat

    for _ in range(max_decisions):
        obs_tensor = trainer.obs_to_agent_input(
            observation,
            agent.device,
            cfg.image_bits,
        )
        action_tensor, prev_state = agent.act(obs_tensor, prev_action, prev_state, action_spec)

        action:np.ndarray = action_tensor.squeeze(0).detach().cpu().numpy().astype(action_spec.dtype)

        # add training noise
        if training:
            noise = np.random.normal(loc=0.0, scale=cfg.exploration_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, action_spec.minimum, action_spec.maximum).astype(action_spec.dtype)

        time_step, reward = trainer.step_with_action_repeat(env, action, cfg.action_repeat)
        observation = trainer.render_pixels(env, cfg.image_size, cfg.camera_id)

        prev_action = torch.from_numpy(action).to(
            device=agent.device, dtype=torch.float32
        ).unsqueeze(0)

        obs.append(observation)
        physics.append(trainer.get_physics_state(env))
        actions.append(action)
        rewards.append(reward)
        if time_step.last():
            break

    obs = np.stack(obs, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    physics = np.stack(physics, axis=0)

    return obs, actions, rewards, physics


def compute_metrics(loss_hist:list[dict[str, float]], num_steps:int, episodes:int, 
                    transitions:int, env_steps:int) -> tuple[dict[str, float | int], str]:
    '''
    Compute the mean losses and generate a metric dict and a string
    '''
    if not loss_hist: 
        raise ValueError("cannot average empty metric history")

    mean_loss = {
        name: sum(loss[name] for loss in loss_hist) / len(loss_hist)
        for name in loss_hist[0]
    }

    metrics = {
        f"loss/{name}": value
        for name, value in mean_loss.items()
    }
    metrics["replay/transitions"] = num_steps
    metrics["train/episodes"] = episodes
    metrics["train/transitions"] = transitions
    metrics["train/env_steps"] = env_steps

    formatted = "\n".join(
        f"{name}={value:.3f}" if isinstance(value, float)
        else f"{name}={value}"
        for name, value in metrics.items()
    )
    return metrics, formatted

@torch.no_grad()
def evaluate_returns(
    cfg: PlaNetTrainConfig | PlaNetDebugConfig,
    env,
    agent: nn.Module,
    *,
    episodes: int,
    seed: int,
) -> list[float]:
    '''
    run one reproducible evaluation without advancing the training RNG streams

    ``env`` must be freshly constructed with ``task_kwargs={"random": seed}``.
    The explicit seed controls stochastic RSSM inference and PlaNet's CEM
    samples; Dreamer uses the actor distribution's approximate mode. Saving
    and restoring RNG state prevents evaluation frequency from changing
    training.
    '''
    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")

    rng_state = checkpointing.collect_rng_states(None, None)
    was_training = agent.training
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent.eval()
        returns = []
        for _ in range(episodes):
            _, _, rewards, _ = get_planet_episode(env, agent, cfg, training=False)
            returns.append(float(np.sum(rewards)))
        return returns
    finally:
        agent.train(was_training)
        checkpointing.restore_rng_states(rng_state, None, None)


def eval_agent(
    cfg: PlaNetTrainConfig | PlaNetDebugConfig,
    env,
    agent: nn.Module,
) -> tuple[dict, str]:
    '''
    Run evaluation episodes and summarize their returns.
    '''
    eval_metric = {}
    episodic_returns = evaluate_returns(
        cfg,
        env,
        agent,
        episodes=cfg.eval_episodes,
        seed=cfg.seed,
    )

    eval_metric["eval/return_mean"] = float(np.mean(episodic_returns))
    eval_metric["eval/return_std"] = float(np.std(episodic_returns))
    eval_metric["eval/return_min"] = float(np.min(episodic_returns))
    eval_metric["eval/return_max"] = float(np.max(episodic_returns))

    formatted = "\n".join(
        f"{name}={value:.3f}" if isinstance(value, float)
        else f"{name}={value}"
        for name, value in eval_metric.items()
    )

    return eval_metric, formatted

@torch.no_grad()
def write_planet_imagination_video(
    cfg: PlaNetTrainConfig | PlaNetDebugConfig,
    env,
    agent: PlaNet,
    step: int
) -> Path:
    '''
    Condition on five frames, imagine up to 45 more, and write both beside reality.
    '''
    rng_state = checkpointing.collect_rng_states(None, None)
    was_training = agent.training
    fps = cfg.video_fps
    output_path = Path(cfg.video_dir) / f"{cfg.domain}_{cfg.task}" / f"seed_{cfg.seed}" / f"step_{step}.mp4"

    agent.eval()
    try:
        obs, actions, _, _ = get_planet_episode(env, agent, cfg, training=False)

        observed_steps = min(5, obs.shape[0])
        imagined_steps = min(45, actions.shape[0] - observed_steps + 1)
        if imagined_steps < 1:
            raise ValueError("evaluation episode is too short for open-loop prediction")

        obs_tensor = torch.cat(
            [
                trainer.obs_to_agent_input(frame, agent.device, cfg.image_bits)
                for frame in obs[:observed_steps]
            ],
            dim=1,
        )
        action_tensor = torch.from_numpy(actions).to(
            device=agent.device,
            dtype=torch.float32,
        )
        actions_into_prefix = torch.cat(
            [
                torch.zeros((1, actions.shape[-1]), device=agent.device),
                action_tensor[:observed_steps - 1],
            ],
            dim=0,
        ).unsqueeze(0)

        embeds = agent.encoder(obs_tensor)
        posterior, _ = agent.rssm.observe(embeds, actions_into_prefix)
        observed_frames = agent.decoder(agent.rssm.get_feat(posterior)).squeeze(0)
        start_state = RSSMState(
            posterior.s_t[:, -1],
            posterior.h_t[:, -1],
            posterior.latent_mean[:, -1],
            posterior.latent_std[:, -1],
        )

        future_actions = action_tensor[
            observed_steps - 1:observed_steps - 1 + imagined_steps
        ].unsqueeze(0)
        imagined_state = agent.rssm.imagine(start_state, future_actions)
        imagined_feat = agent.rssm.get_feat(imagined_state)
        future_frames = agent.decoder(imagined_feat).squeeze(0)
        diagnostic_frames = torch.cat([observed_frames, future_frames], dim=0)

        return trainer.write_side_by_side_video(
            actual_frames=obs[:observed_steps + imagined_steps],
            imagined_frames=diagnostic_frames,
            output_path=output_path,
            fps=fps,
        )
    finally:
        agent.train(was_training)
        checkpointing.restore_rng_states(rng_state, None, None)
    
def train_planet(cfg:PlaNetTrainConfig|PlaNetDebugConfig, resume:str|None=None):
    from dm_control import suite

    '''
    the training loop

    resume: path to a resume checkpoint file or a run's checkpoint dir
    (or the literal "auto" resolved by the CLI). When set, model,
    optimizer, replay buffer, counters, and RNG streams are restored and
    training continues on the same wandb run.
    '''
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = trainer.select_device()
    print(f"Using device: {device}\n")

    # Context managers close both environments when training ends.
    with (
            suite.load(domain_name=cfg.domain,
            task_name=cfg.task,
            task_kwargs={
                "random":cfg.seed
            }
        ) as env, 
            suite.load(domain_name=cfg.domain,
            task_name=cfg.task,
            task_kwargs={
                "random":cfg.seed
            }
        ) as eval_env):

        action_spec = env.action_spec()
        action_dim = action_spec.shape[0]

        agent = PlaNet(
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

        replay = ReplayBuffer(
            capacity=cfg.replay_capacity,
            image_bits=cfg.image_bits,
        )

        checkpoint_dir = Path(cfg.checkpoint_dir) / f"{cfg.domain}_{cfg.task}" / f"seed_{cfg.seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Restore before W&B initialization so logging resumes in the saved run.
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

        # The context manager closes the W&B run when training ends.
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

            # W&B disabled mode may not provide a run ID.
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
                            "agent_type": "planet",
                            "model": agent.state_dict(),
                            "step": step,
                            "env_steps": env_steps_collected,
                            "eval_seed": cfg.seed,
                            "eval_episodes": cfg.eval_episodes,
                            "eval_return": best_eval_return,
                        }, checkpoint_dir/"best.pt")

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
                    "agent_type": "planet",
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

    cfg = PlaNetDebugConfig() if args.debug else PlaNetTrainConfig()
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

    train_planet(cfg, resume=resume)
