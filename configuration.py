from typing import List
import json
import numpy as np


TYPES = {
    'Graph': {
        'graph_path': str,
    },
    'Experiment': {
        'parallel': int,
        'tau': float,
        'victims': List[int],
        'max_enum': int,
        'ratio': float,
        'epoch': List[int],
        'seed': str
    }
}


RANGES = {
    'Experiment': {
        'parallel': [1, np.inf],
        'tau': [0, np.inf],
        'victims': [2, np.inf],
        'max_enum': [0, np.inf],
        'ratio': [0, 1],
        'epoch': [1, np.inf],
    }
}


DEFAULT_INI = {
    'Graph': {
        'graph_path': 'nets'
    },
    'Experiment': {
        'parallel': 1000,
        'tau': 0,
        'victims': [2, 5, 10, 25, 100],
        'max_enum': 2,
        'ratio': .1,
        'epoch': [800],
        "repeat": 10,
        'seed': None
    }
}


class Config(dict):
    """Configuration of a simulation
    """

    def __init__(self, config_path):
        """Init a Config object by reading from file

        Args:
            config_path (str): path to the json config file
        """
        self.config_path = config_path

        with open(config_path, 'r') as f:
            ini = json.load(f)
            self.update(ini)

    def showOptions(self):
        print('config path: ', self.ini_path)
        print(self)
        print(json.dumps(self, indent=4))


if __name__ == '__main__':
    Config()
