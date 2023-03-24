import os
import sys
from pathlib import Path


# get local interpreter path
def get_path() -> Path:
    # get installation folder
    path = Path.home().joinpath('.local/share/ov/pkg/')

    # get all install versions of isaac sim
    files = [f for f in path.glob('isaac_sim-*')]

    # no local isaac installation exists
    if len(files) == 0:
        return None

    # get most up-to-date version of isaac sim
    return files[0]


isaac_path = get_path()


def is_isaac_installed() -> bool:
    return isaac_path is not None


def is_isaac_running() -> bool:
    # isaac tag was added to args when restarting program with isaac interpreter
    return "--isaac" in sys.argv


def start():
    if not is_isaac_installed():
        raise 'No local installation of Isaac exists. Use "Pybullet" environment instead!'

    # get paths of interperter
    python_path = isaac_path.joinpath('python.sh')
    # environment_path = Path('./environment.py').absolute()

    # make sure all requirements are installed in isaac environment
    os.system(f'{python_path} -m pip install -r ./requirements.txt')

    # start environment.py with python interpreter of isaac sim
    command = f'{python_path}'
    for arg in sys.argv:
        command += f' {arg}'

    print(f'Switching python environemnt by running command: {command}')
    os.system(command + ' --isaac')

    # exit old python environment after python interpreter finished execution
    exit(0)


def setup_engine():
    # start program execution with local isaac interpreter
    if not is_isaac_running():
        start()

if __name__ == '__main__':
    start()
