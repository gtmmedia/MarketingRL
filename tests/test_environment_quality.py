import unittest

from marketing_openenv.env import MarketingCampaignEnv
from marketing_openenv.models import Action, Observation, Reward
from marketing_openenv.tasks import TASKS


class EnvironmentQualityTests(unittest.TestCase):
    def test_api_contract_types(self):
        env = MarketingCampaignEnv(task_id="easy_ctr_recovery", seed=123)
        obs = env.reset()
        self.assertIsInstance(obs, Observation)

        next_obs, reward, done, info = env.step(Action(action_type="wait"))
        self.assertIsInstance(next_obs, Observation)
        self.assertIsInstance(reward, Reward)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_task_difficulty_progression(self):
        easy = TASKS["easy_ctr_recovery"]
        medium = TASKS["medium_conversion_push"]
        hard = TASKS["hard_multi_segment_stability"]

        self.assertLess(easy.max_steps, medium.max_steps)
        self.assertLess(medium.max_steps, hard.max_steps)
        self.assertLess(easy.total_budget, medium.total_budget)
        self.assertLess(medium.total_budget, hard.total_budget)
        self.assertLess(easy.target_conversions, medium.target_conversions)
        self.assertLess(medium.target_conversions, hard.target_conversions)

    def test_episode_terminates_and_returns_grade(self):
        env = MarketingCampaignEnv(task_id="easy_ctr_recovery", seed=33)
        env.reset()

        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(Action(action_type="wait"))

        self.assertIn("grade", info)
        self.assertIn("score", info["grade"])
        self.assertGreaterEqual(info["grade"]["score"], 0.0)
        self.assertLessEqual(info["grade"]["score"], 1.0)

    def test_reward_has_partial_progress_components(self):
        env = MarketingCampaignEnv(task_id="medium_conversion_push", seed=99)
        env.reset()
        _, reward, _, _ = env.step(Action(action_type="adjust_bid", channel="search", delta=0.1))

        self.assertIn("progress_delta", reward.components)
        self.assertIn("spend_penalty", reward.components)
        self.assertIn("invalid_penalty", reward.components)


if __name__ == "__main__":
    unittest.main()
