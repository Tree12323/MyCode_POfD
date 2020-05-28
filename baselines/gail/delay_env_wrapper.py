"""
Delay reward wrapper for dense mujoco environment.
"""
import gym


class DelayRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_freq, max_path_length):
        super(DelayRewardWrapper, self).__init__(env)
        # 初始化变量
        self._reward_freq = reward_freq
        self._max_path_length = max_path_length
        self._current_step = 0
        self._delay_step = 0
        self._delay_r_ex = 0.0

    # reset env
    def reset(self):
        obs = self.env.reset()
        self._current_step = 0
        self._delay_step = 0
        self._delay_r_ex = 0.0
        return obs

    def step(self, action):
        # 执行当前act，得到next_obs,reward,done,info
        next_obs, reward, done, info = self.env.step(action)
        # 当前时间步+1
        self._current_step += 1
        # 累计奖赏_delay_r_ex
        self._delay_r_ex += reward
        # 如果当前情节完成1.done，2.大于等于设置的最大情节长度，3.奖赏的长度限制
        if done or self._current_step >= self._max_path_length or self._delay_step == self._reward_freq:
            # 累积奖赏delay_reward
            delay_reward = self._delay_r_ex
            # 重置环境
            self._delay_step = 0
            self._delay_r_ex = 0.0
        else:
            # 正常记录
            delay_reward = 0.0
            self._delay_step += 1
        # 返回next_obs,累积奖赏delay_reward,done,info
        # 如果当前情节结束，返回以上变量，如果没结束，只有next_obs有意义
        return next_obs, delay_reward, done, info