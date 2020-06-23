from ..envs import MA_Gym_Env
from ..utils.train import Experience_Wrapper
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

def multiagent_interact(env, agents, duration=100, record_video=False, video_recording_savepath=None, record_episode_data=True):
    """run simple gym.env rl event loop
    
    args:
        env: envs.MA_Gym_Env to reset/step/close/render
        agents: agents to produce observations and actions
        duration: number of frames to run the environment
        record_video: Boolean. False (default) whether or 
            not to record env interaction video
        video_recording_savepath: location to save env video
            recording at. If `None` (default) will default to
            f'/tmp/single_colony_{str(time.time())}'
        record_episode_data: True (default) to wrap env in
            utils.Experience_Wrapper and return episode for
            training with utils.multiagent_train(episode)

    return: returns the episode if record_episode is True,
        otherwise does not return anything
    """
    if video_recording_savepath is None:
        video_recording_savepath = \
            f'/tmp/multiagent results on {time.asctime()}'

    if record_episode_data:
        env = Experience_Wrapper(env)
    video_recorder = VideoRecorder(env,
        video_recording_savepath,
        enabled=record_video)

    obs_n = env.reset()
    for _ in range(duration):
        a_n = {agent: agent.act(obs)
                for agent, obs in obs_n.items()}
        obs_n, _, _, _ = env.step(a_n)
        video_recorder.capture_frame()

    video_recorder.close()
    if record_episode_data:
        return env.episode()
    else:
        return