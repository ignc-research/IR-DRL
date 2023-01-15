from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes, BaseCallback, EveryNTimesteps
import numpy as np

class MoreLoggingCustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MoreLoggingCustomCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> bool:
        """
        Extracts some values from the envs and logs them. Only works for VecEnvs.
        """
        # success rate
        success_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("success_stat")])
        self.logger.record("train/success_rate_train", success_rate)
        
        # goal metrics (like distance threshold)
        goal_metrics_envs = self.training_env.get_attr("goal_metrics")
        goal_metrics_total = []
        # get all metrics in one list along one dimension
        for env_metrics in goal_metrics_envs:
            for metric in env_metrics:
                goal_metrics_total.append(metric)
        # get all metric names
        goal_names = [metric[0] for metric in goal_metrics_total]
        goal_names = set(goal_names)  # remove duplicates
        # set up a dict for the metrics
        metrics_dict = dict()
        for name in goal_names:
            metrics_dict[name] = []
        for metric in goal_metrics_total:
            metrics_dict[metric[0]].append(metric[1])
        for name in metrics_dict:
            self.logger.record("train/"+name, np.average(metrics_dict[name]))

        # collision rate
        collision_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("collision_stat")])
        self.logger.record("train/collision_rate", collision_rate)

        # timeout rate
        timeout_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("timeout_stat")])
        self.logger.record("train/timeout_rate", timeout_rate)

        # out of bounds rate
        out_of_bounds_rate = np.average([(np.average(ar) if len(ar) != 0 else 0) for ar in self.training_env.get_attr("out_of_bounds_stat")])
        self.logger.record("train/out_of_bounds_rate", out_of_bounds_rate)

        # reward
        rewards_cumulative = np.average(self.training_env.get_attr("reward_cumulative"))
        self.logger.record("train/rewards", rewards_cumulative)
        
        return True

    def _on_step(self) -> bool:
        return True