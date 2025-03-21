import os
import yaml
from .validators import validate_and_log

class Config:
    def __init__(self, cfgpath: str):
        """
        Initialize the configuration object by loading a YAML file,
        validating its content, and setting attributes accordingly.

        Args:
            cfgpath (str): Path to the `config.yaml`.
        """
        with open(cfgpath) as file:
            yaml_conf = yaml.load(file, Loader=yaml.FullLoader)
            validate_and_log(yaml_conf)

        for key, value in yaml_conf.items():
            setattr(self, key, value)

        self._set_paths()

    def _set_paths(self):
        """
        Set configuration paths based on the loaded configuration.
        """
        # Default directories
        self.QGIS_DIR = os.path.join(self.DATA_DIR, "geojson")
        self.IDX_DIR = os.path.join(self.DATA_DIR, "index")
        self.MAP_DIR = os.path.join(self.DATA_DIR, "maps")
        self.RDR_DIR = os.path.join(self.DATA_DIR, "rdr")

        # Default filepaths
        self.TAB_PATH = os.path.join(self.IDX_DIR, "RDRCUMINDEX.TAB")
        self.LBL_PATH = os.path.join(self.IDX_DIR, "RDRCUMINDEX.LBL")