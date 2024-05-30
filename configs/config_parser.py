import os
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Config():
    def __init__(self, cfgpath: str):  
        self.conf = AttrDict()
        with open(cfgpath) as file:
            yaml_cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.conf.update(yaml_cfg)

    def getConfig(self) -> dict:
        """
        Initializes some additional internal variables; based on user input in config.yaml;
        Performs a few tests to ensure that the configuration is valid
        """    
        self.conf.TAB_PATH = f'{self.conf.DATA_DIR}/HiRISE/index/RDRCUMINDEX.TAB'
        self.conf.LBL_PATH = f'{self.conf.DATA_DIR}/HiRISE/index/RDRCUMINDEX.LBL'

        self.conf.QGIS_DIR = f'{self.conf.DATA_DIR}/qgis/qgis_layer'
        self.conf.BBOX_DIR = f'{self.conf.QGIS_DIR}/bbox'
        self.conf.RDR_DIR = f'{self.conf.QGIS_DIR}/rdr'

        return self.conf