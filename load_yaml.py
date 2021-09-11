import yaml

yaml.warnings({'YAMLLoadWarning': False})


class ParameterManager:
    _parameters = {'SpaceGroup': str,
                   'LatticeParameter': float,
                   'Species': list,
                   'Coordinates': list,
                   'MillerIndex': tuple,
                   'SlabThickness': float,
                   'VacuumHeight': float,
                   'supercell': tuple,
                   'z_height': float
                   }

    def __init__(self, filename):
        """
        TODO default value and ctype split!!!

        :param filename:            setting_110.yaml
        """
        self.fname = filename

        self.load()
        self.check_trans()

    def load(self):
        f = open(self.fname, "r", encoding='utf-8')
        cfg = f.read()
        parameters = yaml.load(cfg)
        f.close()
        for key, value in parameters.items():
            setattr(self, key, value)

    def check_trans(self):
        for key, value in ParameterManager._parameters.items():
            if hasattr(self, key) and not isinstance(self.__dict__[key], value):
                if value == tuple:
                    self.__dict__[key] = tuple(eval(self.__dict__[key]))
                else:
                    raise IndexError
