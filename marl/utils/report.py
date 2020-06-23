def multiagent_episode_report(episode):
    """log episode average performance to console
    In the future, this will generate a .csv instead
    to assist intelligence analysis of episode
    
    args:
        episode: multiagent episode

    return: returns nothing"""

    mean_reward = sum([
        sum(list(r_n.values()))/len(r_n)
        for _, _, r_n, _, _ in episode
    ])
    print(f"mean agent reward: " + str(mean_reward))
    return