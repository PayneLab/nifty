import os

class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self, parameters):
        self.parameters = parameters

    def check(self):
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters should be a dictionary.")
        
        for key, value in self.parameters.items():
            if not isinstance(key, str):
                raise ValueError(f"Parameter key '{key}' must be a string.")
            if not isinstance(value, (str, int, float)):
                raise ValueError(f"Parameter value for '{key}' must be a string, int, or float.")
        
        return True