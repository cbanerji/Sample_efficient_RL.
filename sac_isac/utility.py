import gym
import mujoco_py
import numpy as np
import itertools

'''Returns the object type of the observation space, deals with Dicts and arrays only'''
def ret_obs_type(env):
    env = gym.make(env) #initialize environment
    obs = env.reset()
    act = env.action_space.sample()
    typ = None
    #return observation type
    if 'dict' in str(type(obs)):
        typ = 'dict'
    elif 'ndarray' in str(type(obs)):
        typ = 'numparr'
    else:
        print("Unrecognized type")
    #return step length
    step_len = env._max_episode_steps
    return obs, act, step_len, typ


'''Returns the length of observation space and action space vectors'''
def ret_len(env):
    obs, act, step_len, typ = ret_obs_type(env)
    length_lst = None
    if typ == 'dict':
        #count
        obs_samp = obs
        length_dict = {key: len(value) for key, value in obs_samp.items()}
        length_lst = sum(list(length_dict.values())) # Count the observation values
        #return count excluding the key 'Achieved goal'
        length_lst = length_lst-len(obs_samp['achieved_goal'])
    elif typ == 'numparr' :
        length_lst = obs.shape[0]

    act_len = act.shape[0]

    return length_lst,act_len,typ, step_len


'''flattens dict type observation into array type, excluding the 'achieved goal' vector'''
def make_flat(obs,typ):
     if typ == 'dict':
        val_obs = [obs.pop('observation'),obs.pop('desired_goal')]
        obs = list(itertools.chain(*val_obs)) # flatten into array
        obs = np.asarray(obs).reshape(1,len(obs))#----
        return obs
     elif typ == 'numparr' :
        return obs
