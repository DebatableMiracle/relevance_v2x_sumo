# sim/train_ppo_gui_watch.py
import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys
sys.path.insert(0, ".")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gym_env import BroadcastGymEnv

def make_env():
    env = BroadcastGymEnv(
        map_name="circular",
        gui=True,                 # <-- watch it live
        spawn_rate=0.2,
        cav_penetration=0.5,
        broadcast_k=3,
        max_vehicles=25,
        max_steps_per_episode=64,
    )
    return Monitor(env)

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=32,
    verbose=1,
)

model.learn(total_timesteps=3000, log_interval=1)   # short — just enough to watch behavior for a bit
print("✅ GUI watch run complete")