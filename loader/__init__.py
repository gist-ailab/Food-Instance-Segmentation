from .synthetic_loader import *
from .real_loader import *
from .unimib import *

def get_dataset(config, mode='train'):
    """ mode = 'train' or 'val'
    """
    if config["dataset"] == 'synthetic':
        mode = 'val' if mode == 'test' else mode
        dataset = SyntheticDataset(config=config, mode=mode)
    elif config["dataset"] == 'real_tray':
        mode = 'val' if mode == 'test' else mode
        dataset = RealTrayDataset(config=config, mode=mode)
    elif config["dataset"] == 'unimib2016':
        mode = 'test' if mode == 'val' else mode
        dataset = UNIMIB2016Dataset(config=config, mode=mode)
    elif config["dataset"] == 'unimib2016_fake':
        dataset = UNIMIB2016DatasetFake(config=config, mode=mode)
    else:
        raise ValueError("Wrong Dataset Name in CONFIG {}".format(config["dataset"]))
    return dataset
