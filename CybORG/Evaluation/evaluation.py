import inspect
import time
from statistics import mean, stdev
from CybORG import CybORG
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG.Agents.PPO.PPO import PPO
from datetime import datetime

# this imports a submissions agents
from CybORG.Evaluation.submission.submission import wrap


def run_evaluation(name, team, name_of_agent, max_eps, write_to_file=True):

    cyborg_version = 3.1
    scenario = "Scenario3"

    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(sg, "sim")
    wrapped_cyborg = wrap(cyborg)
    agents = {
        f"blue_agent_{agent}": PPO(
            wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0],
            len(wrapped_cyborg.get_action_space(f"blue_agent_{agent}")),
            0.002,
            [0.9, 0.990],
            0.99,
            1,
            0.5,
            True,
            "CybORG/Evaluation/submission/Models/12829.pth",
        )
        for agent in range(18)
    }
    print(
        f"Using agents {agents}, if this is incorrect please update the code to load in your agent"
    )
    if write_to_file:
        file_name = (
            str(inspect.getfile(CybORG))[:-7]
            + "/Evaluation/"
            + time.strftime("%Y%m%d_%H%M%S")
        )
        print(
            f"Saving evaluation results to {file_name}_summary.txt and {file_name}_full.txt"
        )
    start = datetime.now()

    print(f"using CybORG v{cyborg_version}, {scenario}\n")

    total_reward = []
    actions_log = []
    obs_log = []
    for i in range(max_eps):
        observations = wrapped_cyborg.reset()
        action_spaces = wrapped_cyborg.action_spaces
        r = []
        a = []
        o = []
        # cyborg.env.env.tracker.render()
        count = 0
        for j in range(500):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], agent.memory, deterministic=True
                )
                for agent_name, agent in agents.items()
                if agent_name in wrapped_cyborg.agents
            }
            observations, rew, done, info = wrapped_cyborg.step(actions)
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            if write_to_file:
                # a.append({agent_name: str(cyborg.get_last_action(agent_name)) for agent_name in wrapped_cyborg.agents})
                a.append(
                    {
                        agent_name: wrapped_cyborg.get_action_space(agent_name)[
                            actions[agent_name]
                        ]
                        for agent_name in actions.keys()
                    }
                )
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
        total_reward.append(sum(r))
        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)
        if i > 2:
            print(
                f"{i} Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}"
            )
    end = datetime.now()
    difference = end - start
    print(
        f"Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}"
    )
    print(f"file took {difference} amount of time to finish evaluation")
    if write_to_file:
        with open(file_name + "_summary.txt", "w") as data:
            data.write(f"CybORG v{cyborg_version}, {scenario}\n")
            data.write(f"author: {name}, team: {team}, technique: {name_of_agent}\n")
            data.write(
                f"Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}"
            )
            data.write(f"Using agents {agents}")

        with open(file_name + "_full.txt", "w") as data:
            data.write(f"CybORG v{cyborg_version}, {scenario}\n")
            data.write(f"author: {name}, team: {team}, technique: {name_of_agent}\n")
            data.write(
                f"mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n"
            )
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs} \n total reward: {sum_rew}\n"
                )


if __name__ == "__main__":
    # ask for a name
    name = input("Name: ")
    # ask for a team
    team = input("Team: ")
    # ask for a name for the agent
    technique = input("Name of technique: ")
    run_evaluation(name, team, technique, 100)
