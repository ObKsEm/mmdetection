from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class KvBoardDataset(XMLDataset):

    CLASSES = ("KV板-法拉利70年创新合作伙伴", )
