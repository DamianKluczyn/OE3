import json
import os


class Config:
    def __init__(self, config_path=('../configuration/config.json')):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print('Config file error')
            return {}

    def get_param(self, param_path, default=None):
        keys = param_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_param(self, param_path, value):
        keys = param_path.split('.')
        config = self.config or {}
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)
