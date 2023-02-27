import os
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
    # try importing isaac module
    try:
        from omni.isaac.kit import SimulationApp
        return True
    except ImportError:
        return False


def start():
    if not is_isaac_installed():
        raise 'No local installation of ISAAC exists!'

    # get paths of
    python_path = isaac_path.joinpath('python.sh')
    environment_path = Path('./environment.py').absolute()

    # start environment.py with python interpreter of isaac sim
    os.system(f'{python_path} {environment_path}')


def setup_engine():
    # make sure isaac is installed
    if not is_isaac_installed():
        raise 'No local installation of Isaac exists. Use "Pybullet" environment instead!'
    # start program execution with local isaac interpreter
    if not is_isaac_running():
        # restart program with isaac interpreter
        start()
        # exit after completion
        exit(0)

if __name__ == '__main__':
    start()
