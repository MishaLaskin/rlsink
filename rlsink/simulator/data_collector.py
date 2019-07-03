from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np
from tqdm import tqdm

from rlkit.envs.dm import DMGoalPointMassEnv


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

                obs['action'] = any
                obs['image_observation'] = self.env.render(32, 32)
                for k, v in obs.items():
                    data[k].append(v.copy())

        file_name = 'point_mass_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(output_dir + file_name, data)
        print('Saved output in:', output_dir + file_name)


if __name__ == "__main__":
    output_dir = '/home/misha/downloads/vqvae/data/'
    collector = PointMassDataGenerator(path_length=100, num_paths=200)
    collector.run_simulation(output_dir)
    print('done')
