import yaml
from typing import Any
from pathlib import Path
from .validators import validate_and_log

class Config:
    def __init__(self, cfgpath: str):
        """
        Initialize the configuration object by loading a YAML file,
        validating its content, and setting attributes accordingly.

        Args
        ----
            cfgpath : str
                Path to the `config.yaml`.
        """
        with open(cfgpath) as file:
            yaml_conf = yaml.load(file, Loader=yaml.FullLoader)
            validate_and_log(yaml_conf)

        for key, value in yaml_conf.items():
            setattr(self, key, value)

        self._set_paths()

    def __getattr__(self, name: str) -> Any:
        """
        Called when an attribute lookup fails in the normal places
        (i.e. it's not found in __dict__ or via __getattribute__).
        We check our __dict__ and return it to satisfy Pylance.
        """

        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _set_paths(self):
        """
        Set configuration paths based on the loaded configuration.
        """
        # Default directories
        self.DATA_DIR: Path = Path(self.DATA_DIR)
        self.QGIS_DIR: Path = self.DATA_DIR / "geojson"
        self.IDX_DIR: Path = self.DATA_DIR / "index"
        self.MAP_DIR: Path = self.DATA_DIR / "maps"
        self.RDR_DIR: Path = self.DATA_DIR / "rdr"