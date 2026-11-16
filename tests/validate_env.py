from stable_baselines3.common.env_checker import check_env
from environment.racing_env import RacingEnv

def main():
    env = RacingEnv(dt=1.0 / 120.0, max_episode_steps=12000, start_speed=0.0, start_index=1,)

    try:
        print("\nVALIDATING RACING ENVIRONMENT")

        check_env(env, warn=True)

        print("\nENVIRONMENT VALIDATION PASSED")

    finally:
        env.close()


if __name__ == "__main__":
    main()