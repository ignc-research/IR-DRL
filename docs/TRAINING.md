# Configuration Files

Before starting either training or evaluation, it is necessary to pick out a configuration file. In ModularDRLEnv, the actual content of the simulation being run is determined by configuration files written in YAML format. These determine the number and type of robots, tasks, scenarios and sensors as well as other miscellaneous settings.

Most of the scenarios implemented in ModularDRLEnv come with [default configs](../configs/) that instantiate an experiment with hand-picked settings. The [explanation config](../configs/explanations.yaml) shows the range of possible options. In order to add new possibilities to configuration files, please refer to the coding guide (TODO).

# Training/Evaluation

Once you've picked a configuration file, the experiment can be run via ```python run.py <path to configfile> --train|--eval|--debug ```. The ```--train``` option will start a background training process without GUI, ```--eval``` will open a single GUI window while ```--debug``` does the same, but stops the experiment after each step until the user interacts with the console.

When restarting training or trying to evaluate a specific model, the ```load_model``` flag in the config has to be set to true and the model path has to be added as well.