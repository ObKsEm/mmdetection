from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class IconDataset(XMLDataset):

    CLASSES = ("icon", )
