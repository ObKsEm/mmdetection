from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class UavDataset(XMLDataset):

    CLASSES = ('uav', )
