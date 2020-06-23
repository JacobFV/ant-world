# We first get our environment and agents
from ..envs import Grid_World
from ..agents.zoo import Zoo
from ..agents.algorithms.model_based import Model_Based_Algorithm

# The Zoo is a convenience initializer for many agents
# It applies one rl algorithm to many agents
# This algorithm requires an obs and action spec to initialize
# so we just pass the incomplete initializers on to the zoo
# where they can be fully initialized for each agent individually
from functools import partial
zoo = Zoo(partial(Model_Based_Algorithm, discount_factor=0.9))

agents = [zoo.ant()] * 10 + [zoo.caterpillar()] * 20
env = Grid_World(world_size=(16,16), agents=agents)

# Then we alternate interacting and training
from ..utils import multiagent_interact, multiagent_train, multiagent_episode_report

import time
simulation_id = f"{time.asctime()} single_colony"
for epoch in range(10):
    # run the environment interact rollout
    print(f"interacting for epoch{epoch}")
    episode = multiagent_interact(
        env,
        agents,
        duration=100,
        record_video=True,
        video_recording_savepath=
            f"/tmp/{simulation_id}-{epoch}",
        record_episode_data=True)
    # report data for human analysis
    multiagent_episode_report(episode) #quantitative observation
    # train on episode just experienced
    print(f"training for epoch{epoch}")
    multiagent_train(episode) #qualitative observation

print("single colony simulation complete!")