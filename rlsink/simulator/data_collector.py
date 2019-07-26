from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np
from tqdm import tqdm

from rlkit.envs.dm import DMGoalPointMassEnv
from vqvae.envs.reacher import GoalReacher, GoalReacherNoTarget
from vqvae.envs.pusher import GoalPusherNoTarget
from vqvae.envs.utils import SimpleGoalEnv
from vqvae.envs.utils import RefBlockEnv

from rlsink.oracles.path_collector import run_policy


class PointMassDataGenerator:

    def __init__(self,
                 path_length=100,
                 num_paths=1,
                 env_name='point_mass',
                 mode='easy_big'):

        self.path_length = path_length
        self.num_paths = num_paths
        self.env_name = env_name
        self.mode = mode
        self.env = DMGoalPointMassEnv(env_name=self.env_name,
                                      mode=self.mode,
                                      max_steps=self.path_length)

    def run_simulation(self, output_dir=''):
        obs = self.env.reset()
        data = {k: [] for k in obs}
        data['action'] = []
        data['image_observation'] = []

        for i in tqdm(range(self.num_paths)):

            obs = self.env.reset()

            for j in range(self.path_length):
                a = self.env.action_space.sample()
                obs, _, _, _ = self.env.step(a)

                obs['action'] = a
                obs['image_observation'] = self.env.render(32, 32)
                for k, v in obs.items():

                    data[k].append(v.copy())
        file_name = 'point_mass_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(output_dir + file_name, data)
        print('Saved output in:', output_dir + file_name)


class ReacherDataGenerator:

    def __init__(self,
                 path_length=100,
                 num_paths=50,
                 env_name='reacher',
                 mode='easy'):

        self.path_length = path_length
        self.num_paths = num_paths
        self.env_name = env_name
        self.mode = mode
        self.env = GoalReacherNoTarget(max_steps=self.path_length)

    def run_simulation(self, output_dir=''):
        obs = self.env.reset()
        data = {k: [] for k in obs}
        data['action'] = []
        data['image_observation'] = []

        for i in tqdm(range(self.num_paths)):

            obs = self.env.reset()

            for j in range(self.path_length):
                a = self.env.action_space.sample()
                obs, _, _, _ = self.env.step(a)

                obs['action'] = a
                obs['image_observation'] = self.env.dm_env.physics.render(
                    width=32, height=32, camera_id=0)
                for k, v in obs.items():
                    data[k].append(v.copy())
        file_name = 'reacher_no_target_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(output_dir + file_name, data)
        print('Saved output in:', output_dir + file_name)


class PusherDataGenerator:

    def __init__(self,
                 path_length=100,
                 num_paths=20):

        self.path_length = path_length
        self.num_paths = num_paths
        self.env = SimpleGoalEnv(obs_dim=42, goal_dim=3, env_name='stacker', reward_type='sparse',
                                 task='push_1', max_steps=self.path_length)

    def run_simulation(self, output_dir=''):
        obs = self.env.reset()
        data = {k: [] for k in obs}
        data['image_observation'] = []
        dir_ = '/home/misha/research/rlkit/data'
        file_name = '/pusher-real-sparse-SACHER-jul24/pusher_real_sparse_SACHER_jul24_2019_07_24_22_56_45_0000--s-0/params.pkl'

        render_kwargs = dict(width=64, height=64, camera_id=0)
        for i in tqdm(range(self.num_paths)):
            # each iteration do a random or learned policy
            if i % 2 == 0:
                obs = self.env.reset()

                for j in range(self.path_length):
                    a = self.env.action_space.sample()
                    obs, _, _, _ = self.env.step(a)

                    obs['image_observation'] = self.env.dm_env.physics.render(
                        **render_kwargs)
                    for k, v in obs.items():
                        data[k].append(v.copy())
            else:
                # run learned policy
                path, _ = run_policy(dir_+file_name, self.env, goal_env=True,
                                     use_color=False, cherrypick=True, fixed_length=True, verbose=True, render_kwargs=render_kwargs)
                for j, item in enumerate(path["observations"]):
                    # to keep policy unbiased flip the images every other time
                    item["image_observation"] = path["images"][j]

                    for k, v in item.items():
                        data[k].append(v.copy())

        file_name = 'pusher_no_target_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(output_dir + file_name, data)
        print('Saved output in:', output_dir + file_name)


class BlockDataGenerator:

    def __init__(self,
                 path_length=100,
                 num_paths=20):

        self.path_length = path_length
        self.num_paths = num_paths
        self.env = RefBlockEnv()

    def run_simulation(self, output_dir=''):
        self.env.reset()
        data = {}
        data['image_observation'] = []

        render_kwargs = dict(width=64, height=64, camera_id=0)
        for i in tqdm(range(self.num_paths)):
            # each iteration do a random or learned policy
            if i % 10 == 0:
                self.env.reset()
                if np.random.randint(2):
                    self.env.set_block_pos(x=.5, z=.2)
                else:
                    self.env.set_block_pos(x=-.5, z=.2)

                for j in range(self.path_length):
                    a = np.random.uniform(-1, 1,
                                          self.env.dm_env.action_spec().shape)
                    self.env.step(a)
                    img = self.env.dm_env.physics.render(
                        **render_kwargs)
                    data['image_observation'].append(img)
            else:
                self.env.reset()
                for _ in range(self.path_length):
                    self.env.set_block_pos(x=np.random.uniform(-.37, .37))
                    img = self.env.dm_env.physics.render(
                        **render_kwargs)
                    data['image_observation'].append(img)

        file_name = 'single_block_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(output_dir + file_name, data)
        print('Saved output in:', output_dir + file_name)


if __name__ == "__main__":
    output_dir = '/home/misha/research/vqvae/data/'
    collector = BlockDataGenerator(path_length=100, num_paths=100)
    collector.run_simulation(output_dir)
    print('done')
