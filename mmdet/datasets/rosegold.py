from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class RoseGoldDataset(XMLDataset):

    CLASSES = (
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-20 4L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-20 1L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-30 4L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-30 1L',
        '壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 4L',
        '壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 1L',
        '壳牌先锋超凡喜力 ACEA C2 / C3 天然气全合成油 0W-30 4L',
        '壳牌先锋超凡喜力 ACEA C2 / C3 天然气全合成油 0W-30 1L',
        "壳牌先锋超凡喜力 ACEA A3 / B4 天然气全合成油 0W-40 4L",
        "壳牌先锋超凡喜力 ACEA A3 / B4 天然气全合成油 0W-40 1L",
        '其他',
    )


@DATASETS.register_module
class RoseGoldMidDataset(XMLDataset):

    CLASSES = (
        '550053925',
        '550053924',
        '550053927',
        '550053926',
        '550053918',
        '550053917',
        '550053915',
        '550053914',
        '000000002',
        '000000001',
        'Unknown',
    )
