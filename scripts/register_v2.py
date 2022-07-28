from copy import deepcopy

import gym
from d4rl.locomotion import maze_env
from gym.envs.registration import register


def register_new_env():
    template_kwargs = dict(
        id=None,
        entry_point='d4rl.locomotion.ant:make_ant_maze_env',
        max_episode_steps=1000,
        kwargs={
            'maze_map': None,
            'reward_type': 'sparse',
            'dataset_url': None,
            'non_zero_reset': False,
            'eval': True,
            'maze_size_scaling': 4.0,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0,
            'v2_resets': True,
        }
    )

    list_env_info = [
        (
            'antmaze-medium-noisy-v2',
            maze_env.HARDEST_MAZE_TEST,
            'https://storage.googleapis.com/heterogeneous_data/recollect_antmaze/Antmaze_hardest-maze_biased_True_multigoal_False_new_larger_bias_relabelsparse.hdf5'
        ),
        (
            'antmaze-medium-biased-v2',
            maze_env.HARDEST_MAZE_TEST,
            'https://storage.googleapis.com/heterogeneous_data/recollect_antmaze/Antmaze_hardest-maze_biased_True_multigoal_False_new_actually_larger_bias_relabel_final_large_biassparse.hdf5'
        ),

        (
            'antmaze-large-noisy-v2',
            maze_env.BIG_MAZE_TEST,
            'https://storage.googleapis.com/heterogeneous_data/recollect_antmaze/Antmaze_big-maze_biased_True_multigoal_False_new_larger_bias_relabelsparse.hdf5'
        ),
        (
            'antmaze-large-biased-v2',
            maze_env.BIG_MAZE_TEST,
            'https://storage.googleapis.com/heterogeneous_data/recollect_antmaze/Antmaze_big-maze_biased_True_multigoal_False_new_actually_larger_bias_relabel_final_large_biassparse.hdf5'
        )
    ]

    for env_info in list_env_info:

        id, maze_map, dataset_url = env_info

        kw = deepcopy(template_kwargs)

        kw['id'] = id
        kw['kwargs']['maze_map'] = maze_map
        kw['kwargs']['dataset_url'] = dataset_url

        register(**kw)

    return list_env_info


def make_all_env():

    list_env_info = register_new_env()

    for env_info in list_env_info:

        env_name, *_ = env_info

        gym.make(env_name)
