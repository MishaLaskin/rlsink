from dm_control import suite
import numpy as np


class Oracle:

    def __init__(self,
                 env_name=None,
                 task=None,
                 path_length=100,
                 img_dim=32,
                 camera_id=0,
                 n_paths=50):
        self.env_name = env_name
        self.task = task
        self.path_length = path_length
        self.n_paths = n_paths
        self.img_dim = img_dim
        self.camera_id = camera_id

        self.dm_env = suite.load(env_name, task)
        self.spec = self.dm_env.action_spec()

    def generate_path(self, strategy='random_normal'):
        path = getattr(self, strategy + '_strategy')()
        return path

    def random_normal_strategy(self):

        self.dm_env = stationary_reset(self.dm_env)
        path = self.init_path()
        for _ in range(self.path_length):
            a = rand_a(self.spec)
            ts = self.dm_env.step(a)
            obs = ts.observation
            img = self.render()
            path = self.update_path(path, a, obs, img)

        return path

    def update_path(self, path, a, obs, img):
        path["actions"].append(a)
        path["observations"].append(obs)
        path["images"].append(img)
        return path

    def init_path(self):
        return dict(images=[], observations=[], actions=[])

    def render(self):
        return self.dm_env.physics.render(width=self.img_dim,
                                          height=self.img_dim, camera_id=self.camera_id)


class PusherOracle(Oracle):

    def __init__(self,
                 env_name='stacker',
                 task='stack_1',
                 path_length=100,
                 img_dim=32,
                 camera_id=0,
                 n_paths=50):

        super().__init__(env_name=env_name,
                         task=task,
                         path_length=path_length,
                         img_dim=img_dim,
                         camera_id=camera_id,
                         n_paths=n_paths)

    def random_reasonable_strategy(self):

        self.dm_env = stationary_reset(self.dm_env)
        path = self.init_path()
        for _ in range(self.path_length):
            a = rand_a(self.spec)
            ts = self.dm_env.step(a)
            obs = ts.observation
            img = self.render()
            path = self.update_path(path, a, obs, img)

        return path


def rand_a(spec):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


def stationary_reset(env, n_steps=200):
    env.reset()
    spec = env.action_spec()
    for _ in range(n_steps):
        action = rand_a(spec)
        env.step(action)

    return env


oracle = PusherOracle()
path = oracle.generate_path(strategy='random_normal')
print(path["images"][0].shape, len(path["images"]))
