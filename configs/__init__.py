from .config_parser import Config
CONF = Config('configs/configs.yaml').getConfig()

__all__ = ['CONF']