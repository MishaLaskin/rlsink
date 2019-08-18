from vqvae.envs.reacher import GoalVQVAEEnv
import numpy as np
import pickle
import networkx as nx


class GoalReacherNoTargetVQVAE(GoalVQVAEEnv):
    def __init__(self,
                 obs_dim=None,
                 goal_dim=None,
                 model_path=None,
                 graph_path=None,
                 img_dim=32,
                 z_dim=128,
                 camera_id=0,
                 gpu_id=0,
                 reward_type='sparse',
                 rep_type='continuous',
                 max_steps=500,
                 threshold=0.15,
                 explore=False,
                 **kwargs):
        # be sure to specify obs dim and goal dim

        super().__init__(env_name='reacher', mode='no_target',
                         obs_dim=obs_dim, act_dim=None, goal_dim=goal_dim,
                         model_path=model_path,
                         img_dim=img_dim,
                         camera_id=camera_id,
                         gpu_id=gpu_id)
        obj = pickle.load(
            open("/home/misha/research/rlsink/saved/reacher_graph.pkl", "rb"))
        self.graph, self.nodes = obj['graph'], obj['nodes'].reshape(
            -1, obs_dim)

        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.reward_type = reward_type
        self.desired_goal = None
        self.rep_type = rep_type
        self.current_rep = None
        self.last_rep = None
        self.explore = explore
        self.rep_counts = {}
        self.plan = None
        self.z_dim = z_dim

    def generate_plan(self, obs_dict):
        start_distances = np.linalg.norm(
            self.nodes-obs_dict['achieved_goal'], axis=1)
        end_distances = np.linalg.norm(
            self.nodes-obs_dict['desired_goal'], axis=1)
        start_id = np.argmin(start_distances)
        end_id = np.argmin(end_distances)
        node_ids = nx.dijkstra_path(self.graph, start_id, end_id)

        path_nodes = self.nodes[node_ids]
        # node_ds = np.array([np.linalg.norm(path_nodes[i]-path_nodes[i+1])
        #                    for i in range(len(node_ids)-1)])

        #important_ids = np.argwhere(node_ds > self.threshold).reshape(-1)

        #node_ids = list(np.array(node_ids)[important_ids+1])

        return list(node_ids)

    def compute_reward(self, action, obs, *args, **kwargs):
        # abstract method only cares about obs and threshold

        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        if self.reward_type == 'sparse':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'dense':
            r = - distance
        else:
            raise_error(err_type='reward')

        return r

    def compute_rewards(self, actions, obs):
        # abstract method only cares about obs and threshold
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type == 'sparse':
            r = -(distances > self.threshold).astype(float)
        elif self.reward_type == 'dense':
            r = -distances.astype(float)
        else:
            raise_error(err_type='reward')
        return r

    def is_done(self, obs):
        # abstract method only cares about obs and threshold
        # check if max step limit is reached
        if len(self.plan) == 0:
            return True, True

        if self.steps >= self.max_steps:
            done = True
            is_success = False
            return done, is_success

        # check if episode was successful

        distance = np.linalg.norm(
            obs['achieved_goal'] - self.nodes[self.plan[-1]])
        is_success = distance < self.threshold
        done = is_success

        return done, is_success

    def step(self, a):
        obs_dict = self.step_and_get_obs_dict(a)
        #obs_dict['desired_goal'] = self.nodes[self.plan[0]]
        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(obs_dict)

        current_node_id = self.node_from_obs(obs_dict)
        current_node = self.nodes[current_node_id]
        self.plan = self.advance_plan(obs_dict)

        info = {
            'is_success': is_success,
            'fraction_completed': (float(self.path_length)-float(len(self.plan)))/float(self.path_length),
            'num_reps': len(self.visited_reps),
            'latent_distance': np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal']),
            'rep_counts': len(self.rep_counts.keys()),
            'node_id': current_node_id,
            'node': current_node
        }

        return obs_dict, reward, done, info

    def node_from_obs(self, obs_dict):
        d = np.linalg.norm(self.nodes - obs_dict["observation"], axis=1)
        node_id = np.argmin(d)
        return node_id

    def step_and_get_obs_dict(self, action):
        assert self.encoded_goal is not None, "Must set desired goal before stepping"

        # 1. step forward with action
        # 2. get image, encode it

        _ = self.dm_env.step(action)

        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash)

        goal = self.nodes[self.plan[0]] if len(
            self.plan) else self.encoded_goal
        obs_dict = self.construct_obs_dict(
            z_e, z_e, goal, rep_hash, self.goal_rep_hash)

        return obs_dict

    def encode_observation(self, is_goal=False, include_indices=True):
        if is_goal:
            img = self.reset_goal_image()
            self.goal_img = img
        else:
            img = self.get_current_image()

        img = self.numpy_to_tensor_img(img)
        img = self.normalize_image(img)

        z_e = self.encode_image(img, as_tensor=True)

        if include_indices:
            _, _, _, _, e_indices = self.model.vector_quantization(z_e)
            e_indices = self.normalize_indices(
                e_indices).detach().cpu().numpy()
            return z_e.reshape(-1).detach().cpu().numpy(), e_indices.reshape(-1)

        return z_e.reshape(-1).detach().cpu().numpy(), None

    def reset(self):
        valid_plan = False
        while not valid_plan:
            # generate goal, encode it, get continuous and discrete values
            self.encoded_goal, self.goal_indices = self.encode_observation(
                is_goal=True, include_indices=True)

            self.goal_rep_hash = hash(tuple(self.goal_indices))
            # don't reset env again, since that will generate target in different position
            # get observation encoding
            z_e, e_indices = self.encode_observation(
                is_goal=False, include_indices=True)

            rep_hash = hash(tuple(e_indices))

            self.update_internal_state(rep_hash, reset=True)

            obs_dict = self.construct_obs_dict(
                z_e, z_e, self.encoded_goal, rep_hash, self.goal_rep_hash)

            # generate plan

            self.plan = self.generate_plan(obs_dict)

            self.path_length = len(self.plan)

            if self.path_length > 1:
                valid_plan = True

        # get rid of first node since we're already there
        if True:

            self.plan = self.plan[:2]

        self.original_plan = self.plan.copy()
        self.start_node_id = self.plan.pop(0)

        # get the subgoal
        subgoal = self.nodes[self.plan[0]]
        obs_dict['desired_goal'] = subgoal
        self.encoded_goal = self.nodes[self.plan[-1]]

        # basically makes policy just get to nearest node

        self.end_node_id = self.plan[-1]
        return obs_dict

    def advance_plan(self, obs_dict):
        all_subgoal_nodes = self.nodes[self.plan]

        proximities = np.linalg.norm(
            all_subgoal_nodes - obs_dict['achieved_goal'], axis=1)

        min_ids = np.argwhere(proximities < self.threshold).reshape(-1)
        # print(proximities)
        # print('min_ids', min_ids, self.threshold,
        #      np.mean(obs_dict['achieved_goal']))
        if len(min_ids) > 0:
            min_id = np.max(min_ids)
            plan = self.plan[min_id+1:]
        else:
            plan = self.plan
        return plan

    def reset_goal_image(self):
        """
        Resets the environment and generates goal images
        """
        # reset to get goal image
        self.dm_env.reset()
        goal_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        goal_img = goal_img.reshape(1, *goal_img.shape)
        self.goal_img = goal_img.copy()

        # reset to get different starting image
        self.dm_env.reset()
        return goal_img

    def normalize_image(self, img):
        """normalizes image to [-1,1] interval

        Arguments:
            img {np.array or torch.tensor} -- [an image array / tensor with integer values 0-255]

        Returns:
            [np.array or torch tensor] -- [an image array / tensor with float values in [-1,1] interval]
        """
        # takes to [0,1] interval
        img /= 255.0
        # takes to [-0.5,0.5] interval
        img -= 0.5
        # takes to [-1,1] interval
        img /= 0.5
        return img

    def normalize_indices(self, x):
        dim = 128.0
        assert max(x) < dim, 'index mismatch during normalization'
        x = x.float()
        x /= (dim-1)
        x -= 0.5
        x /= 0.5
        return x

    def update_internal_state(self, rep_hash, reset=False):
        if reset:
            self.steps = 0
            self.visited_reps = set()
            self.visited_reps.add(rep_hash)
            self.current_rep = rep_hash
            self.last_rep = 0
        else:
            self.steps += 1
            self.visited_reps.add(rep_hash)
            self.last_rep = self.current_rep
            self.current_rep = rep_hash
        # increment count in rep_counts
        if rep_hash in self.rep_counts:
            self.rep_counts[rep_hash] += 1
        else:
            self.rep_counts[rep_hash] = 1

    def construct_obs_dict(self, obs, achieved_goal, desired_goal, achieved_rep_hash, desired_rep_hash):
        obs_dict = dict(
            observation=obs,
            state_observation=obs,
            achieved_goal=achieved_goal,
            state_achieved_goal=achieved_rep_hash,
            desired_goal=desired_goal,
            state_desired_goal=desired_rep_hash
        )
        return obs_dict


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')


if __name__ == "__main__":
    env = GoalReacherNoTargetVQVAE(
        obs_dim=128,
        goal_dim=128,
        model_path='/home/misha/research/vqvae/results/vqvae_data_reacher_aug7_ne128nd2.pth',
        graph_path='/home/misha/research/rlsink/saved/reacher_graph.pkl',
        img_dim=32,
        camera_id=0,
        gpu_id=0,
        reward_type='sparse',
        rep_type='continuous',
        max_steps=500,
        threshold=0.15,
        explore=False)

    obs = env.reset()
    # print(obs)
    # a = env.action_space.sample()
    # obs_, r, d, info = env.step(a)
    # print(r, d, info)
    obj = pickle.load(
        open("/home/misha/research/rlsink/saved/reacher_graph.pkl", "rb"))
    G, nodes = obj['graph'], obj['nodes']

    x1 = np.linalg.norm(
        obj["nodes"].reshape(-1, 128)-obs["observation"], axis=1)
    x2 = np.linalg.norm(
        obj["nodes"].reshape(-1, 128)-obs["desired_goal"], axis=1)
    all_nodes = env.nodes[env.plan]
    x2 = np.linalg.norm(
        all_nodes-obs["observation"], axis=1)

    x3 = [np.linalg.norm(all_nodes[i]-all_nodes[i+1])
          for i in range(len(all_nodes)-1)]
    i = np.argmin(x1)
    print(x1[i], x3)
    print(env.plan)
    for j in range(1000):
        a = env.action_space.sample()
        obs_, r, d, info = env.step(a)
        final_goal = env.nodes[env.plan[-1]]
        if j % 50 == 0:
            print(np.linalg.norm(obs_["achieved_goal"]-final_goal))
    print(env.plan)
