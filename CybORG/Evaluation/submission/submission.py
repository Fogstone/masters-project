from CybORG.Agents.Wrappers import PettingZooParallelWrapper

# from CybORG.Agents.PPO.PPO import PPO

# agents = {f"blue_agent_{agent}": PPO() for agent in range(18)} #Agents initialized in evaluation due to parameters


def wrap(env):
    return PettingZooParallelWrapper(env=env)


submission_name = "Masters"
submission_team = "Fog"
submission_technique = "PPO"
