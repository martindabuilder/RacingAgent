from pathlib import Path
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback, CallbackList,)
from stable_baselines3.common.monitor import Monitor
from environment.racing_env import RacingEnv

TRACK_NAMES = ["yas_marina", "hockenheim", "sochi", "sepang"]
TOTAL_TIMESTEPS = 500_000
MAX_EPISODE_STEPS = 12000
MODEL_NAME = "ppo_racing_agent"
CHECKPOINT_FREQUENCY = 100_000
PROGRESS_FREQUENCY = 10_000
RANDOM_SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_DIR / "training"
MODELS_DIR = TRAINING_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = TRAINING_DIR / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


def format_time(seconds):
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, report_frequency=10000):
        super().__init__(verbose=0)

        self.total_timesteps = total_timesteps
        self.report_frequency = report_frequency

        self.start_time = None
        self.next_report = report_frequency

    def _on_training_start(self):

        self.start_time = time.perf_counter()

    def _on_step(self):

        current_steps = self.num_timesteps

        if current_steps >= self.next_report or current_steps >= self.total_timesteps:

            elapsed = time.perf_counter() - self.start_time

            progress = min(
                current_steps / self.total_timesteps,
                1.0
            )

            steps_per_second = (
                current_steps / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining_steps = max(
                0,
                self.total_timesteps - current_steps
            )

            estimated_remaining = (
                remaining_steps / steps_per_second
                if steps_per_second > 0
                else 0.0
            )

            print(
                f"\rProgress: {progress * 100:6.2f}%"
                f" | Steps: {current_steps:,}/{self.total_timesteps:,}"
                f" | Speed: {steps_per_second:.1f} steps/s"
                f" | Elapsed: {format_time(elapsed)}"
                f" | Remaining: {format_time(estimated_remaining)}",
                end="",
                flush=True
            )

            self.next_report += self.report_frequency

        return True

    def _on_training_end(self):

        print("\n")


def create_training_environment():
    env = RacingEnv(
        track_names=TRACK_NAMES,
        max_episode_steps=MAX_EPISODE_STEPS,
        start_speed=0.0,
    )

    env = Monitor(env)
    return env


def main():

    print("\nTraining tracks:")
    for track_name in TRACK_NAMES:
        print(f"- {track_name}")

    print(f"\nTotal timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Maximum episode steps: {MAX_EPISODE_STEPS:,}")
    print(f"Checkpoint frequency: {CHECKPOINT_FREQUENCY:,}")

    env = create_training_environment()

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(CHECKPOINTS_DIR),
        name_prefix=MODEL_NAME,
    )

    progress_callback = TrainingProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        report_frequency=PROGRESS_FREQUENCY,
    )

    callback = CallbackList([
        checkpoint_callback,
        progress_callback,
    ])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },

        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        device="auto",
        seed=RANDOM_SEED,
    )

    print(f"\nTraining device: {model.device}")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name=MODEL_NAME,
        )

        final_model_path = MODELS_DIR / MODEL_NAME

        model.save(str(final_model_path))

        print("\nTraining complete, final model saved to:")
        print(final_model_path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted_model_path = (MODELS_DIR / f"{MODEL_NAME}_interrupted")
        model.save(str(interrupted_model_path))
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()

if __name__ == "__main__":
    main()