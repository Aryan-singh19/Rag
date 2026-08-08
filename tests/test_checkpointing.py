import numpy as np
import torch
import pytest

from worldmodels.config import (
    DreamerDebugConfig,
    DreamerTrainConfig,
    PlaNetDebugConfig,
    PlaNetTrainConfig,
)
from worldmodels.utils import ReplayBuffer
from worldmodels.trainer import checkpointing
from worldmodels import trainer
from worldmodels.eval import build_agent


def make_fake_episode(T=12, action_dim=2, state_dim=8):
    obs = np.random.randint(0, 256, size=(T + 1, 64, 64, 3), dtype=np.uint8)
    actions = np.random.uniform(-1, 1, size=(T, action_dim)).astype(np.float32)
    rewards = np.random.uniform(0, 1, size=(T,)).astype(np.float32)
    physics = np.random.uniform(-1, 1, size=(T + 1, state_dim)).astype(np.float64)
    return obs, actions, rewards, physics


def test_replay_state_dict_roundtrip():
    src = ReplayBuffer(capacity=100, image_bits=5)
    for _ in range(3):
        src.add_episode(*make_fake_episode())

    dst = ReplayBuffer(capacity=100, image_bits=5)
    dst.load_state_dict(src.state_dict())

    assert dst.num_steps == src.num_steps
    assert len(dst.buffer) == len(src.buffer)
    for ep_src, ep_dst in zip(src.buffer, dst.buffer):
        assert torch.equal(ep_src["obs"], ep_dst["obs"])
        assert torch.equal(ep_src["actions"], ep_dst["actions"])
        assert torch.equal(ep_src["rewards"], ep_dst["rewards"])
        assert torch.equal(ep_src["physics"], ep_dst["physics"])

    # restored buffer must be sampleable
    obs, actions, rewards = dst.sample_batch(batch_size=2, seq_len=4)
    assert obs.shape[0] == 2


def test_atomic_save_replaces(tmp_path):
    target = tmp_path / "ckpt.pt"
    checkpointing.atomic_torch_save({"v": 1}, target)
    checkpointing.atomic_torch_save({"v": 2}, target)
    payload = torch.load(target, weights_only=False)
    assert payload["v"] == 2
    assert not (tmp_path / "ckpt.pt.tmp").exists()


def test_milestones_crossed():
    milestones = (100, 250, 500)
    assert checkpointing.milestones_crossed(milestones, 0, 99) == []
    assert checkpointing.milestones_crossed(milestones, 99, 100) == [100]
    assert checkpointing.milestones_crossed(milestones, 90, 260) == [100, 250]
    assert checkpointing.milestones_crossed(milestones, 250, 501) == [500]
    assert checkpointing.milestones_crossed((), 0, 10**9) == []
    assert PlaNetTrainConfig().milestone_env_steps == (500_000, 1_000_000)
    assert DreamerTrainConfig().milestone_env_steps == (500_000, 1_000_000)
    assert PlaNetDebugConfig().milestone_env_steps == ()
    assert DreamerDebugConfig().milestone_env_steps == ()


def test_snapshot_keeps_exact_pixels_with_physics():
    replay = ReplayBuffer(capacity=100, image_bits=5)
    replay.add_episode(*make_fake_episode())
    original_obs = replay.buffer[0]["obs"].clone()

    state = replay.state_dict()
    assert "obs" in state["episodes"][0]
    assert "physics" in state["episodes"][0]

    restored = ReplayBuffer(capacity=100, image_bits=5)
    restored.load_state_dict(state)
    assert torch.equal(restored.buffer[0]["obs"], original_obs)
    obs, actions, rewards = restored.sample_batch(batch_size=2, seq_len=4)
    assert obs.shape[0] == 2


def test_snapshot_keeps_pixels_for_physicsless_episodes():
    replay = ReplayBuffer(capacity=100, image_bits=5)
    obs, actions, rewards, _ = make_fake_episode()
    replay.add_episode(obs, actions, rewards)

    state = replay.state_dict()
    assert "obs" in state["episodes"][0], "no physics means pixels must be kept"


def test_pixelless_checkpoint_is_rejected_before_replay_mutation():
    source = ReplayBuffer(capacity=100, image_bits=5)
    source.add_episode(*make_fake_episode())
    state = source.state_dict()
    state["episodes"] = [
        {key: value for key, value in episode.items() if key != "obs"}
        for episode in state["episodes"]
    ]

    destination = ReplayBuffer(capacity=100, image_bits=5)
    destination.add_episode(*make_fake_episode(T=6))
    original_obs = destination.buffer[0]["obs"].clone()

    with pytest.raises(ValueError, match="omits replay pixels"):
        destination.load_state_dict(state)
    assert torch.equal(destination.buffer[0]["obs"], original_obs)


def test_mps_rng_state_roundtrip(monkeypatch):
    saved_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored_states = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: saved_state.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", restored_states.append)

    states = checkpointing.collect_rng_states(object(), object())
    checkpointing.restore_rng_states(states, object(), object())

    assert torch.equal(states["torch_mps"], saved_state)
    assert len(restored_states) == 1
    assert torch.equal(restored_states[0], saved_state)


def test_seeded_evaluation_is_repeatable_and_preserves_rng(monkeypatch):
    from worldmodels.trainer import planet_trainer

    cfg = PlaNetDebugConfig(eval_episodes=3)

    class FakeAgent:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    class FakeEnv:
        def __init__(self, seed):
            self.task = type("Task", (), {"random": np.random.RandomState(seed)})()

    def fake_episode(env, agent, cfg, training):
        assert not training
        reward = env.task.random.random() + np.random.random() + torch.rand(()).item()
        return None, None, np.array([reward]), None

    monkeypatch.setattr(planet_trainer, "get_planet_episode", fake_episode)
    agent = FakeAgent()

    with pytest.raises(ValueError, match="at least one episode"):
        planet_trainer.evaluate_returns(cfg, None, agent, episodes=0, seed=7)

    np.random.seed(123)
    torch.manual_seed(456)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    first = planet_trainer.evaluate_returns(
        cfg, FakeEnv(7), agent, episodes=cfg.eval_episodes, seed=7,
    )

    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert agent.training

    np.random.random(10)
    torch.rand(10)
    second = planet_trainer.evaluate_returns(
        cfg, FakeEnv(7), agent, episodes=cfg.eval_episodes, seed=7,
    )
    assert first == second

    def failing_episode(env, agent, cfg, training):
        np.random.random()
        torch.rand(())
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(planet_trainer, "get_planet_episode", failing_episode)
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        planet_trainer.evaluate_returns(cfg, FakeEnv(7), agent, episodes=1, seed=7)
    numpy_after = np.random.get_state()
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert agent.training


def test_video_diagnostic_failure_preserves_rng_and_mode(monkeypatch, tmp_path):
    from worldmodels.trainer import planet_trainer

    cfg = PlaNetDebugConfig(video_dir=str(tmp_path))

    class FakeAgent:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    def failing_episode(env, agent, cfg, training):
        np.random.random()
        torch.rand(())
        raise RuntimeError("video failed")

    monkeypatch.setattr(planet_trainer, "get_planet_episode", failing_episode)
    agent = FakeAgent()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="video failed"):
        planet_trainer.write_planet_imagination_video(cfg, object(), agent, step=1)

    numpy_after = np.random.get_state()
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert agent.training


def test_dmc_resume_checkpoint_roundtrip(tmp_path):
    from dm_control import suite

    cfg = PlaNetDebugConfig()
    device = torch.device("cpu")

    with (
        suite.load(cfg.domain, cfg.task, task_kwargs={"random": cfg.seed}) as env,
        suite.load(cfg.domain, cfg.task, task_kwargs={"random": cfg.seed}) as eval_env,
    ):
        action_dim = env.action_spec().shape[0]
        agent = build_agent(cfg, action_dim, device)
        replay = ReplayBuffer(capacity=cfg.replay_capacity, image_bits=cfg.image_bits)
        replay.add_episode(*trainer.collect_random_episode(env, cfg))

        # one real update so the optimizer has state worth checkpointing
        obs, actions, rewards = replay.sample_batch(cfg.batch_size, cfg.sequence_length)
        agent.update(obs, actions, rewards)

        path = checkpointing.save_resume_checkpoint(
            tmp_path, cfg, agent, replay, env, eval_env,
            step=7,
            episodes_collected=1,
            transitions_collected=replay.num_steps,
            env_steps_collected=replay.num_steps * cfg.action_repeat,
            best_eval_return=12.5,
            wandb_run_id="test-run-id",
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert "obs" in saved["replay"]["episodes"][0]

        # note the RNG position right after saving, then disturb everything
        rng_probe = np.random.uniform()
        param_ref = [p.detach().clone() for p in agent.parameters()]
        with torch.no_grad():
            for p in agent.parameters():
                p.add_(1.0)
        np.random.uniform(size=1000)

        agent2 = build_agent(cfg, action_dim, device)
        replay2 = ReplayBuffer(capacity=cfg.replay_capacity, image_bits=cfg.image_bits)
        payload = checkpointing.load_resume_checkpoint(
            tmp_path / checkpointing.RESUME_FILENAME, agent2, replay2, env, eval_env,
        )

        assert payload["step"] == 7
        assert payload["best_eval_return"] == 12.5
        assert payload["wandb_run_id"] == "test-run-id"
        assert replay2.num_steps == replay.num_steps

        # replay pixels are stored exactly, including task-randomized scenes
        assert torch.equal(replay2.buffer[0]["obs"], replay.buffer[0]["obs"])

        for p_ref, p_loaded in zip(param_ref, agent2.parameters()):
            assert torch.equal(p_ref, p_loaded)

        # optimizer state came back (exp_avg exists for some param)
        opt_state = agent2.optimizer.state_dict()["state"]
        assert len(opt_state) > 0

        # RNG stream resumes from the save point, not from where the
        # disturbance left it
        assert np.random.uniform() == pytest.approx(rng_probe)
