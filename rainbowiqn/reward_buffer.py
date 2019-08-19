import numpy as np
from collections import deque


# This class handle storing all rewards at specific time: 5 min, 30min and infinite (SABER)
# This is used only by actor 0 and it serves for plotting the reward curve along time step
# with max, min and standard deviation over 100 episodes.
# This is also used to dump all scores and episode length in a csv file
class RewardBuffer:
    def __init__(self, evaluation_episodes, action_repeat):

        # We initialize all buffer with 0 because sometimes there are not totally
        # filled for the first evaluation step and this leads to a bug in the plot...

        # This buffer stores the SABER score along the last episodes
        self.total_reward_buffer_SABER = deque(
            [0] * evaluation_episodes, maxlen=evaluation_episodes
        )
        # This buffer stores the score along the last episodes with a max length episode of 30min
        self.total_reward_buffer_30min = deque(
            [0] * evaluation_episodes, maxlen=evaluation_episodes
        )
        # This buffer stores the score along the last episodes with a max length episode of 5min
        self.total_reward_buffer_5min = deque(
            [0] * evaluation_episodes, maxlen=evaluation_episodes
        )
        # This buffer stores the episode length along the last episodes
        self.episode_length_buffer = deque([0] * evaluation_episodes, maxlen=evaluation_episodes)

        # This is the reward of the current episode without any time cap (SABER)
        self.current_total_reward_SABER = 0
        # This is the reward of the current episode with a 30 min time cap
        # (i.e. we stop adding reward if we reach 30 minutes length)
        self.current_total_reward_30min = 0
        # This is the reward of the current episode with a 5 min time cap
        # (i.e. we stop adding reward if we reach 5 minutes length)
        self.current_total_reward_5min = 0

        # This is a tab containing the total steps done by all actors (x axis of the reward plot)
        self.Tab_T_actors = []
        # This is a tab containing the step done by learner (overlay in the reward plot)
        self.Tab_T_learner = []
        # This is a tab containing the mean length of episode (overlay in the reward plot)
        self.Tab_mean_length_episode = []
        # This is a tab containing the maximum length episode
        # along with his score (overlay in the reward plot)
        self.Tab_longest_episode = []
        # This is a list of list containing all the rewards encountered during all evaluation steps
        # i.e. self.tab_rewards_plot[i] contains the 100 scores obtained during the i-th evaluation
        self.tab_rewards_plot = []
        # This is the best average reward encountered while training to keep the best model
        self.best_avg_reward = -1e10
        self.action_repeat = action_repeat

    def update_score_episode_buffer(self, timestep):
        self.total_reward_buffer_SABER.append(self.current_total_reward_SABER)

        # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
        if timestep < (5 * 60 * 60) / self.action_repeat:
            self.current_total_reward_5min = self.current_total_reward_SABER
        self.total_reward_buffer_5min.append(self.current_total_reward_5min)

        # 30 minutes * 60 secondes * 60 HZ Atari game / action repeat
        if timestep < (30 * 60 * 60) / self.action_repeat:
            self.current_total_reward_30min = self.current_total_reward_SABER
        self.total_reward_buffer_30min.append(self.current_total_reward_30min)

        self.episode_length_buffer.append(timestep)
        self.current_total_reward_SABER = 0
        self.current_total_reward_30min = 0
        self.current_total_reward_5min = 0

    def update_current_reward_buffer(self, timestep, reward):
        # THIS should be before clipping, we want to know the true score of the game there!
        self.current_total_reward_SABER += reward
        # 5 minutes * 60 secondes * 60 HZ Atari game / action repeat
        if timestep == (5 * 60 * 60) / self.action_repeat:
            self.current_total_reward_5min = self.current_total_reward_SABER
        # 30 minutes * 60 secondes * 60 HZ Atari game / action repeat
        if timestep == (30 * 60 * 60) / self.action_repeat:
            self.current_total_reward_30min = self.current_total_reward_SABER

    def update_step_actors_learner_buffer(self, T_total_actors, T_learner):
        self.Tab_T_actors.append(T_total_actors)
        self.Tab_T_learner.append(T_learner)

        current_avg_episode_length = sum(self.episode_length_buffer) / len(
            self.episode_length_buffer
        )
        self.Tab_mean_length_episode.append(current_avg_episode_length)

        indice_longest_episode = np.argmax(self.episode_length_buffer)
        self.Tab_longest_episode.append(
            (
                self.episode_length_buffer[indice_longest_episode],
                self.total_reward_buffer_SABER[indice_longest_episode],
            )
        )

        return sum(self.total_reward_buffer_SABER) / len(self.total_reward_buffer_SABER)
