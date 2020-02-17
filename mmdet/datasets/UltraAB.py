from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class UltraABDataset(XMLDataset):

    CLASSES = (
        '壳牌恒护超凡喜力欧系专属 5W-30 1L',
        '壳牌恒护超凡喜力欧系专属 5W-30 4L',
        '壳牌恒护超凡喜力欧系专属 5W-40 1L',
        '壳牌恒护超凡喜力欧系专属 5W-40 4L',
        '壳牌恒护超凡喜力亚系专属 5W-30 1L',
        '壳牌恒护超凡喜力亚系专属 5W-30 4L',
        '其他'
    )


@DATASETS.register_module
class UltraABMidDataset(XMLDataset):

    CLASSES = (
        '550055084',
        '550055085',
        '550055152',
        '550055153',
        '550055114',
        '550055115',
        'Unknown',
    )
