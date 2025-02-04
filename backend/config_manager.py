from database import get_config, set_config

class ConfigManager:
    @staticmethod
    def get(key):
        return get_config(key)

    @staticmethod
    def set(key, value):
        set_config(key, value)

    @staticmethod
    def get_all():
        return get_all_config()
