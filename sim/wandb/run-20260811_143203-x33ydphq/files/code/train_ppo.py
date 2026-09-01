import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys
sys.path.insert(0, ".")

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gym_env import BroadcastGymEnv

config = {
    "algo": "PPO",
    "total_timesteps": 100_000,
    "cav_penetration": 0.5,
    "broadcast_k": 5,
    "spawn_rate": 0.4,
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
}

run = wandb.init(
    project="v2x-broadcast-selection",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

def make_env():
    env = BroadcastGymEnv(
        config="grid.sumocfg", net_file="../net/grid.net.xml",
        gui=False, spawn_rate=config["spawn_rate"],
        cav_penetration=config["cav_penetration"],
        broadcast_k=config["broadcast_k"],
        max_steps_per_episode=500,
    )
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    verbose=1,
    tensorboard_log=f"../logs/tb/{run.id}",
)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"../models/{run.id}",
        verbose=2,
    ),
)

model.save(f"../models/{run.id}/final_model")
vec_env.save(f"../models/{run.id}/vecnormalize.pkl")

run.finish()
print("✅ Training complete")