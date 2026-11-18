from pathlib import Path
import shutil
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback, CallbackList,)
from stable_baselines3.common.monitor import Monitor
from environment.racing_env import RacingEnv


TRACK_NAMES = ["yas_marina", "hockenheim", "sochi", "sepang", "monza", "sao_paulo", "shanghai",]

TOTAL_TIMESTEPS = 5_000_000 #4m. secondary retrain
MAX_EPISODE_STEPS = 24000
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

        self.total_timesteps = int(total_timesteps)
        self.report_frequency = int(report_frequency)

        self.start_time = None
        self.start_steps = 0
        self.next_report = self.report_frequency

    def _on_training_start(self):
        self.start_time = time.perf_counter()
        self.start_steps = self.num_timesteps

    def _on_step(self):
        current_steps = self.num_timesteps - self.start_steps

        if (current_steps >= self.next_report or current_steps >= self.total_timesteps):
            elapsed = time.perf_counter() - self.start_time
            progress = min(current_steps / self.total_timesteps, 1.0)

            steps_per_second = (
                current_steps / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining_steps = max(0, self.total_timesteps - current_steps)
            estimated_remaining = (
                remaining_steps / steps_per_second
                if steps_per_second > 0
                else 0.0
            )

            print(f"\rProgress: {progress * 100:6.2f}% | Steps: {current_steps:,}/{self.total_timesteps:,} | Speed: {steps_per_second:.1f} steps/s | Elapsed: {format_time(elapsed)} | Remaining: {format_time(estimated_remaining)}", end="", flush=True)
            self.next_report += self.report_frequency

        return True

    def _on_training_end(self):
        print("\n")


def create_training_environment():
    env = RacingEnv(track_names=TRACK_NAMES, max_episode_steps=MAX_EPISODE_STEPS, start_speed=0.0,)
    env = Monitor(env)
    return env


def load_or_create_model(env):
    model_path = MODELS_DIR / f"{MODEL_NAME}.zip"

    if model_path.exists():
        print("\nExisting PPO model found:")
        print(model_path)

        backup_path = MODELS_DIR / f"{MODEL_NAME}_before_finetune.zip"

        if not backup_path.exists():
            shutil.copy2(model_path, backup_path)
            print("\nOriginal model backup created:")
            print(backup_path)

        print("\nLoading existing model for continued training...")

        model = PPO.load(str(model_path), env=env, device="auto",)
        return model, True

    print("\nNo existing model found.")
    print("Creating a new PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256],}},
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
        device="auto",
        seed=RANDOM_SEED,
    )

    return model, False


def main():
    print("\nTraining tracks:")
    for track_name in TRACK_NAMES:
        print(f"- {track_name}")

    print(f"\nAdditional timesteps: {TOTAL_TIMESTEPS:,}, Maximum episode steps: {MAX_EPISODE_STEPS:,}, Checkpoint frequency: {CHECKPOINT_FREQUENCY:,}")
    env = create_training_environment()

    try:
        model, resume_training = load_or_create_model(env)
        print(f"\nTraining device: {model.device}")
        checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQUENCY, save_path=str(CHECKPOINTS_DIR), name_prefix=MODEL_NAME,)

        progress_callback = TrainingProgressCallback(total_timesteps=TOTAL_TIMESTEPS, report_frequency=PROGRESS_FREQUENCY,)
        callback = CallbackList([checkpoint_callback, progress_callback,])

        print("\nTraining mode:")

        if resume_training:
            print("- Continuing from existing model.")
        else:
            print("- Training a new model.")

        print(f"- Additional steps: {TOTAL_TIMESTEPS:,}")

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=MODEL_NAME, reset_num_timesteps=not resume_training,)
        final_model_path = MODELS_DIR / MODEL_NAME
        model.save(str(final_model_path))

        print("\nTraining complete")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupted_model_path = (MODELS_DIR / f"{MODEL_NAME}_interrupted")
        model.save(str(interrupted_model_path))
        print("\nModel saved to:")
        print(interrupted_model_path)

    finally:
        env.close()

if __name__ == "__main__":
    main()