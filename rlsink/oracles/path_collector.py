import pickle
import time
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.envs.wrappers import NormalizedBoxEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.rollout_functions import multitask_rollout_visualizer, rollout_visualizer
import numpy as np
from vqvae.envs.pusher import EasyPusher, GoalPusher, GoalPusherNoTarget


dir_ = '/home/misha/research/rlkit/data'
file = '/goal-pusher-SAC-HER-VQVAE-jul20/goal_pusher_SAC-HER-VQVAE_jul20_2019_07_20_18_42_57_0000--s-0/params.pkl'


def run_policy(file,
               eval_env,
               goal_env=False,
               use_color=True,
               cherrypick=False,
               fixed_length=False,
               verbose=False,
               render_kwargs=dict(
                   height=128, width=128,
                   camera_id=0
               )):

    ptu.set_gpu_mode(True, 0)

    with open(file, 'rb') as f:
        params = pickle.load(f)

    if goal_env:
        obs_dim = eval_env.observation_space.spaces['observation'].low.size
        action_dim = eval_env.action_space.low.size
        goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    else:
        obs_dim = eval_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size

    policy = params['exploration/policy']  # .to(ptu.device)
    policy = policy.eval()
    policy = MakeDeterministic(policy)
    if goal_env:
        r = [-1]
        step = 0
        while 0 not in r or sum(r) == 0:
            step += 1
            start = time.time()
            if goal_env:
                path = multitask_rollout_visualizer(eval_env,
                                                    agent=policy,
                                                    max_path_length=eval_env.max_steps,
                                                    render=True,
                                                    render_kwargs=render_kwargs,
                                                    observation_key='observation',
                                                    desired_goal_key='desired_goal',
                                                    get_action_kwargs=None,
                                                    return_dict_obs=True,
                                                    use_color=use_color,
                                                    fixed_length=fixed_length
                                                    )

                r = path["rewards"]

            else:
                path = rollout_visualizer(eval_env,
                                          agent=policy,
                                          max_path_length=eval_env.max_steps,
                                          render=True,
                                          render_kwargs=render_kwargs,
                                          use_color=use_color)

                r = path["rewards"]
            if verbose:
                print(step, len(r), sum(r), end='\r')
            if not cherrypick:
                break

    return path, eval_env


if __name__ == "__main__":
    env = GoalPusherNoTarget(max_steps=100, threshold=0.05)
    path, env = run_policy(dir_+file, env, goal_env=True,
                           use_color=False, cherrypick=True, fixed_length=True, verbose=True)

    r = path['rewards']
    print(len(r), sum(r))
    print(r[-5:])
    print(np.array(path["images"]).shape)
    print('DONE')
