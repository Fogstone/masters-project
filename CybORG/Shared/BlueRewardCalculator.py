from collections import namedtuple

from CybORG.Shared import Scenario
from CybORG.Shared.RedRewardCalculator import DistruptRewardCalculator, PwnRewardCalculator
from CybORG.Shared.RewardCalculator import RewardCalculator


HostReward = namedtuple('HostReward','confidentiality availability')

class ConfidentialityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on compromise of hosts/data
    def __init__(self, team_name: str, scenario: Scenario, adversary):
        self.scenario = scenario
        self.adversary = adversary
        super(ConfidentialityRewardCalculator, self).__init__(team_name)
        self.infiltrate_rc = PwnRewardCalculator(self.adversary, scenario)
        self.compromised_hosts = {}
        self.patched_hosts = {}

    def reset(self):
        self.infiltrate_rc.reset()
        self.patched_hosts = {}

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.compromised_hosts = {}
        self.patched_hosts = {}
        reward = -self.infiltrate_rc.calculate_reward(current_state, action, agent_observations, done)
        
        # Calculate additional reward for patching vulnerabilities
        for hostname, host_info in current_state.items():
            if hostname == 'success':
                continue
                
            # Check if the host has patched vulnerabilities
            if 'patched_vulnerabilities' in host_info and 'drone_comms_8888' in host_info['patched_vulnerabilities']:
                # Reward is based on the same confidentiality value used for compromising
                if hostname not in self.patched_hosts:
                    value = self.scenario.get_host(hostname).get_confidentiality_value('Low')
                    mapping = {'None': 0.0, 'Low': 0.5, 'Medium': 2.0, 'High': 5.0}
                    patch_reward = mapping[value]
                    self.patched_hosts[hostname] = patch_reward
                    reward += patch_reward
        
        self._calculate_compromised_hosts()
        return reward

    def _calculate_compromised_hosts(self):
        for host, value in self.infiltrate_rc.compromised_hosts.items():
            self.compromised_hosts[host] = -1 * value


class AvailabilityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on reduction in availability
    def __init__(self, team_name: str, scenario: Scenario, adversary):
        super(AvailabilityRewardCalculator, self).__init__(team_name)
        self.adversary = adversary
        self.disrupt_rc = DistruptRewardCalculator(self.adversary, scenario)
        self.impacted_hosts = {}

    def reset(self):
        self.disrupt_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.impacted_hosts = {}
        reward = -self.disrupt_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_impacted_hosts()
        return reward

    def _calculate_impacted_hosts(self):
        for host, value in self.disrupt_rc.impacted_hosts.items():
            self.impacted_hosts[host] = -1 * value

class HybridAvailabilityConfidentialityRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality reward calculator
    def __init__(self, team_name: str, scenario: Scenario, adversary):
        super(HybridAvailabilityConfidentialityRewardCalculator, self).__init__(team_name)
        self.availability_calculator = AvailabilityRewardCalculator(team_name, scenario, adversary)
        self.confidentiality_calculator = ConfidentialityRewardCalculator(team_name, scenario, adversary)

    def reset(self):
        self.availability_calculator.reset()
        self.confidentiality_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = self.availability_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.confidentiality_calculator.calculate_reward(current_state, action, agent_observations, done)
        self._compute_host_scores(current_state.keys())
        return reward

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.confidentiality_calculator.compromised_hosts
        impacted_hosts = self.availability_calculator.impacted_hosts
        patched_hosts = getattr(self.confidentiality_calculator, 'patched_hosts', {})
        
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            patched = patched_hosts[host] if host in patched_hosts else 0
            
            # Include patching in the reward calculation
            reward_state = HostReward(compromised + patched, impacted)
            self.host_scores[host] = reward_state
