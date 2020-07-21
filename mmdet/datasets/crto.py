from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class CRTODataset(XMLDataset):

    CLASSES = (
        '壳牌劲霸 k4',
        '壳牌劲霸 k6',
        '壳牌劲霸 k8',
        '壳牌劲霸 k10',
        '壳牌劲霸 k15',
        '其他',
    )


@DATASETS.register_module
class CRTOMidDataset(XMLDataset):

    CLASSES = (
        '550055760',
        '550052349',
        '550052346',
        '550052352',
        '550054823',
        'Unknown',
    )
