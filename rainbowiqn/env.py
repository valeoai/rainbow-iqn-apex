from collections import deque
import atari_py
import math

import cv2  # Note that importing cv2 before torch may cause segfaults?

import numpy as np


class Env:
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", args.seed)
        self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
        self.ale.setFloat("repeat_action_probability", args.proba_sticky_actions)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        # ROM loading must be done after setting options
        self.ale.loadROM(atari_py.get_game_path(args.game))
        actions = self.ale.getLegalActionSet()  # We always use 18 actions. See revisiting ALE.
        self.actions = {i: e for i, e in zip(range(len(actions)), actions)}
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.action_repeat = args.action_repeat

        self.SABER_mode = not args.disable_SABER_mode
        if self.SABER_mode:
            # We need to divide time by action repeat to get the max_step_stuck
            self.max_step_stuck_SABER = int(args.max_frame_stuck_SABER / self.action_repeat)

        # Reward on defender are really weird at the time we write this code (7 August 2019).
        # All rewards are basically multiplied by 100 and there is always a initial
        # reward of 10 both for no reason
        if args.game == "defender":
            self.handle_bug_in_defender = True
        else:
            self.handle_bug_in_defender = False

    def _get_state(self):
        np_state_uint8 = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR
        )
        return np_state_uint8

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(np.zeros((84, 84)))

    def reset(self):
        # Reset internals
        self._reset_buffer()
        self.ale.reset_game()

        # Process and return "initial" state
        np_state_uint8 = self._get_state()

        self.state_buffer.append(np_state_uint8)

        if self.SABER_mode:
            self.SABER_mode_count = 0

        return list(self.state_buffer)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros((2, 84, 84), dtype=np.uint8)
        reward, done = 0, False
        for t in range(self.action_repeat):
            reward += self.ale.act(self.actions.get(action))
            if t == self.action_repeat - 2:
                frame_buffer[0] = self._get_state()
            elif t == self.action_repeat - 1:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = np.max(frame_buffer, axis=0)
        self.state_buffer.append(observation)

        # In SABER mode we track how much time without receiving any rewards
        # If stuck for more than 5 minutes, we end the episode
        if self.SABER_mode:
            if reward == 0:
                self.SABER_mode_count += 1
            else:
                self.SABER_mode_count = 0

            if self.SABER_mode_count > self.max_step_stuck_SABER:
                # We didn't receive any reward for 5 minutes, probably game stuck.
                # Let's end this episode
                done = True

        # HANDLING BUG ON REWARD, particularly on defender!
        # This is specific to defender
        if self.handle_bug_in_defender:
            if reward < -1:
                reward = 1000000 + reward  # Buffer rollover
            if reward < 100:
                reward = 0  # We always got a initial reward of 10 for no reason, let's set it to 0
            else:
                reward = math.ceil(reward / 100)

        # HANDLING buffer rollover (happen at least on VideoPinball, Defender and Asterix)
        if reward < -900000:
            print(
                "We got a reward inferior to -900000 this is almost certainly a buffer rollover, "
                "this bug was spotten only on VideoPinball, Defender and Asterix for the moment"
            )
            reward = 1000000 + reward

        # Return state, reward, done
        return list(self.state_buffer), reward, done

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow("screen", self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
