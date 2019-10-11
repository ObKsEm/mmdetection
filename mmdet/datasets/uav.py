from .voc import VOCDataset
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class UavDataset(XMLDataset):

    CLASSES = ('uav', )
