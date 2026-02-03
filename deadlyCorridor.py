from vizdoom import DoomGame
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import cv2
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

class VizDoomGym(Env):
    def __init__(self, render=False, config='defend_the_center_s3.cfg', frame_skip=4, frame_stack=4):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config)
        
        self.game.set_window_visible(render)
        self.game.init()
        
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = []
        
        self.observation_space = Box(low=0, high=1, shape=(frame_stack,100, 160), dtype=np.float32)
        self.actions = self.game.get_available_buttons_size()
        self.action_space = Discrete(self.actions)          
        
        self.health = 100
        self.ammo = 52
        self.killcount = 0
        self.last_x = 0
        self.last_z = 0
        if 'deadly_corridor' in config:
            self.reward_function = self.deadly_corridor_reward
        elif 'defend_the_center' in config:
            self.reward_function = self.defend_center_reward
        elif 'deathmatch' in config:
            self.reward_function = self.deathmatch_reward
        else:
            print(f"Warning: No specific reward found for {config}, using generic.")
            self.reward_function = self.deadly_corridor_reward
        
    def step(self, action):
        total_reward = 0
        done = False
        
        for _ in range(self.frame_skip):
            reward = self.game.make_action(self._action_to_doom(action))
            total_reward += reward
            
            if self.game.is_episode_finished():
                done = True
                break

        if not done:
            state = self._get_observation()

            game_vars = self.game.get_state().game_variables

            reward_del_juego = self.reward_function(game_vars)

            total_reward += reward_del_juego

            self.health = game_vars[0]
            self.killcount = game_vars[1]
            self.ammo = game_vars[2]
            self.last_x = game_vars[3]
            if game_vars[4]:
                self.last_z = game_vars[4]

        else:
            state = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            'health': self.health,
            'killcount': self.killcount,
            'ammo': self.ammo,
            'position': [self.last_x, self.last_z]
        }
        
        return state, total_reward, done, False, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self.frames = []
        
        obs = self._get_observation()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        
        return self._stack_frames(), {}
    
    def _get_observation(self):
            if self.game.is_episode_finished():
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            state = self.game.get_state().screen_buffer
            gray = cv2.cvtColor(np.moveaxis(state, 0, -1), cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            
            # formato (channels, height, width)
            return np.expand_dims(normalized, axis=0)  

    def _stack_frames(self):
        return np.concatenate(self.frames[-self.frame_stack:], axis=0)
    
    def _action_to_doom(self, action):
        actions = np.identity(self.actions)
        return actions[action]

    def deadly_corridor_reward(self, game_vars):
        # Desempaquetar variables
        health, damage_taken, killcount, ammo, x_pos = game_vars

        # Calcular deltas
        health_reward = -0.5 * max(0, self.health - health)  # Penalizar pérdida de vida
        kill_reward = (killcount - self.killcount) * 500.0  # Premiar kills

        progress_reward = 10 * (x_pos - self.last_x) if x_pos > self.last_x else 0

        ammo_pickup = 2 if ammo > self.ammo else 0
        out_of_ammo = -0.2 if (ammo == 0 and self.ammo > 0) else 0

        # Penalización fuerte por morir
        done_penalty = -100 if health <= 0 else 0

        # Sumar todo
        step_reward = (
                health_reward +
                kill_reward +
                progress_reward +
                ammo_pickup +
                out_of_ammo +
                done_penalty
        )

        # Actualiza el estado interno para la siguiente iteración
        self.health = health
        self.killcount = killcount
        self.ammo = ammo
        self.last_x = x_pos

        return step_reward
    
    def defend_center_reward(self, game_vars):
        
        health, damage_taken, killcount, ammo, x_pos = game_vars  
        
        d_ammo = ammo - self.ammo
        if d_ammo == 0:
            ammo_r = 0
        elif d_ammo > 0:
            ammo_r = d_ammo * 0.5
        else:
            ammo_r = -d_ammo * 0.5
        return ammo_r

    def deathmatch_reward(self, game_vars):

        health, damage_taken, killcount, ammo, x_pos, z_pos = game_vars
        shaped_reward = 0

        delta_kills = killcount - self.killcount
        shaped_reward += delta_kills * 200

        delta_health = health - self.health
        shaped_reward += delta_health * 1.0

        if health == 0:
            shaped_reward -= 50

        delta_ammo = ammo - self.ammo
        shaped_reward += delta_ammo * 0.3

        delta_dist = np.linalg.norm([x_pos - self.last_x, z_pos - self.last_z])
        shaped_reward += delta_dist * 0.1

        if delta_kills == 0 and delta_health == 0 and delta_ammo == 0 and delta_dist < 0.5:
            shaped_reward -= 0.05

        self.health = health
        self.ammo = ammo
        self.killcount = killcount
        self.last_x = x_pos
        self.last_z = z_pos


        return shaped_reward

    
    def close(self):
        self.game.close()

class StageAwareCallback(BaseCallback):
    def __init__(self, check_freq, save_path, stage_info, eval_env, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.stage_info = stage_info
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Guardar modelo periódicamente
            model_path = os.path.join(
                self.save_path, 
                f"stage_{self.stage_info['id']}_step_{self.n_calls}"
            )
            self.model.save(model_path)
            
            # Evaluar el modelo
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=10,
                deterministic=True
            )
            
            # Guardar si es el mejor modelo hasta ahora
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
            
            # Loggear métricas
            self.logger.record("stage/mean_reward", mean_reward)
            self.logger.record("stage/stage_id", self.stage_info['id'])
            self.logger.record("stage/difficulty", self.stage_info['difficulty'])
            
            if self.verbose > 0:
                print(f"Stage {self.stage_info['id']} - Step {self.n_calls}:")
                print(f"Mean reward: {mean_reward:.2f}")
        
        return True

def make_env(config, rank=0, seed=0, render=False):
    def _init():
        env = VizDoomGym(config=config, render=render)
        env = Monitor(env)
        np.random.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train():
    # Configuración de etapas con curriculum learning
    stages = [
        {'id': 1, 'config': 'deadly_corridor_s1_killcount.cfg', 'timesteps': 300000, 'difficulty': 2},
        {'id': 2, 'config': 'deadly_corridor_s2_killcount.cfg', 'timesteps': 400000, 'difficulty': 3},
        {'id': 3, 'config': 'deadly_corridor_s3_killcount.cfg', 'timesteps': 500000, 'difficulty': 4},
        {'id': 4, 'config': 'deadly_corridor_s4_killcount.cfg', 'timesteps': 600000, 'difficulty': 5},
        {'id': 5, 'config': 'deadly_corridor_s5_killcount.cfg', 'timesteps': 800000, 'difficulty': 6}
    ]
    stages_center = [
        {'id': 0, 'config': 'defend_the_center_s1.cfg', 'timesteps': 300000, 'difficulty': 1},
        {'id': 1, 'config': 'defend_the_center_s2.cfg', 'timesteps': 400000, 'difficulty': 2},
        {'id': 2, 'config': 'defend_the_center_s3.cfg', 'timesteps': 500000, 'difficulty': 3},
    ]
    stages_line = [
        {'id': 0, 'config': 'defend_the_line_s1.cfg', 'timesteps': 300000, 'difficulty': 1},
        {'id': 1, 'config': 'defend_the_line_s2.cfg', 'timesteps': 400000, 'difficulty': 2},
        {'id': 2, 'config': 'defend_the_line_s3.cfg', 'timesteps': 500000, 'difficulty': 3},
    ]
    
    stages_deathmatch = [
        {'id': 0, 'config': 'deathmatch_s1.cfg', 'timesteps': 500000, 'difficulty': 1},
        {'id': 1, 'config': 'deathmatch_s2.cfg', 'timesteps': 600000, 'difficulty': 2},
        {'id': 2, 'config': 'deathmatch_s3.cfg', 'timesteps': 700000, 'difficulty': 3},
    ]
    
    # Directorios para logs y modelos
    folder_sufix = "deadly_corridor"
    LOG_DIR = "./Doom_RL/logs/log_corridor_curriculum_"+ folder_sufix
    MODEL_DIR = "./Doom_RL/models/"+folder_sufix
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Hiperparámetros del modelo
    policy_kwargs = {
        'normalize_images':False,
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True
    }
    
    model = None
    for stage in stages:
        print(f"\nStarting Stage {stage['id']} (Difficulty: {stage['difficulty']})")
        
        # Crear entornos
        train_env = DummyVecEnv([make_env(stage['config']) for _ in range(4)])
        eval_env = DummyVecEnv([make_env(stage['config'], render=False)])
        
        # Crear modelo o cargar el anterior
        if model is None:
            model = PPO(
                'CnnPolicy',
                train_env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                learning_rate=2.5e-4,
                n_steps=8192,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs
            )
        else:
            model.set_env(train_env)
            model.learning_rate = max(model.learning_rate * 0.8, 1e-6)
        
        # Callbacks
        stage_callback = StageAwareCallback(
            check_freq=10000,
            save_path=os.path.join(MODEL_DIR, f"stage_{stage['id']}"),
            stage_info=stage,
            eval_env=eval_env
        )
        
        # Entrenamiento
        model.learn(
            total_timesteps=stage['timesteps'],
            callback=stage_callback,
            tb_log_name=f"stage_{stage['id']}"
        )
        
        # Guardar modelo completado
        model.save(os.path.join(MODEL_DIR, f"stage_{stage['id']}_completed"))
        
        # Limpieza
        train_env.close()
        eval_env.close()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    train()