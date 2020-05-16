from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class RZXDataset(XMLDataset):

    CLASSES = (
        'like2',
        'account',
        'password',
        'login',
        'like1',
        'search',
        'Not Now',
        'sad2',
        'like',
        'love',
        'haha',
        'wow',
        'sad',
        'angry'
    )



