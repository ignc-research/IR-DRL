# Configuration Files

Before starting either training or evaluation, it is necessary to pick out a configuration file. In ModularDRLEnv, the actual content of the simulation being run is determined by configuration files written in YAML format. These determine the number and type of robots, tasks, scenarios and sensors as well as other miscellaneous settings.

Most of the scenarios implemented in ModularDRLEnv come with [default configs](../configs/) that instantiate an experiment with hand-picked settings. The [explanation config](../configs/explanations.yaml) shows the range of possible options. In order to add new possibilities to configuration files, please refer to the coding guide (TODO).

# Training/Evaluation

Once you've picked a configuration file, the experiment can be run via ```python run.py <path to configfile> --train|--eval|--debug ```. The ```--train``` option will start a background training process without GUI, ```--eval``` will open a single GUI window while ```--debug``` does the same, but stops the experiment after each step until the user interacts with the console.

When restarting training or trying to evaluate a specific model, the ```load_model``` flag in the config has to be set to true and the model path has to be added as well.

ModularDRLEnv offers the possibility of logging a range of data both onto the console and in a separate CSV file. In order to enable this, change the ```logging``` option in the config file to one of the following options:

- 0, no logging
- 1, logging onto the console at the end of an episode
- 2, same as 1 + all steps are logged into a CSV file which is either
  - saved to models/env_logs/ after each episode, in case ```max_episodes``` in the config is set to -1
  - or saved to the same spot after the last episode has run, in case ```max_episodes``` is set to any positive value, the CSV will then also contain data from all episodes
- 3, same as 2 + detailled logging about non-robot objects in the experiment (this greatly increases the size of the log file)