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

    def run_simulation(self):
        obs = self.env.reset()
        self.data = {k: [] for k in obs}
        self.data['action'] = []
        self.data['image_observation'] = []

        for i in tqdm(range(self.num_paths)):

            obs = self.env.reset()
            for j in range(self.path_length):
                a = self.env.action_space.sample()
                obs, _, _, _ = self.env.step(a)
                obs['action'] = a
                obs['image_observation'] = self.env.render(32, 32)
                for k, v in obs.items():
                    self.data[k].append(v)

        file_name = 'point_mass_length' + \
            str(self.path_length) + '_paths_' + str(self.num_paths)
        np.save(file_name, self.data)
        self.data = {}


if __name__ == "__main__":
    collector = PointMassDataGenerator(path_length=100, num_paths=2000)
    collector.run_simulation()
    print('done')
