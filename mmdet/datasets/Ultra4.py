from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class Ultra4Dataset(XMLDataset):

    CLASSES = (
        '壳牌恒护超凡喜力欧系专属 5W-30 1L',
        '壳牌恒护超凡喜力欧系专属 5W-30 4L',
        '壳牌恒护超凡喜力欧系专属 5W-40 1L',
        '壳牌恒护超凡喜力欧系专属 5W-40 4L',
        '壳牌恒护超凡喜力亚系专属 5W-30 1L',
        '壳牌恒护超凡喜力亚系专属 5W-30 4L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-20 4L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-20 1L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-30 4L',
        '壳牌先锋超凡喜力 SN PLUS 天然气全合成油 0W-30 1L',
        '壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 4L',
        '壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 1L',
        '壳牌先锋超凡喜力 ACEA C2 / C3 天然气全合成油 0W-30 4L',
        '壳牌先锋超凡喜力 ACEA C2 / C3 天然气全合成油 0W-30 1L',
        '壳牌先锋超凡喜力 ACEA A3 / B4 天然气全合成油 0W-40 4L',
        '壳牌先锋超凡喜力 ACEA A3 / B4 天然气全合成油 0W-40 1L',
        '壳牌超凡喜力系列 1L',
        '壳牌超凡喜力系列 4L',
        '壳牌极净超凡喜力系列 1L',
        '壳牌极净超凡喜力系列 4L',
        '其他'
    )


@DATASETS.register_module
class Ultra4MidDataset(XMLDataset):

    CLASSES = (
        '550055084',
        '550055085',
        '550055152',
        '550055153',
        '550055114',
        '550055115',
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
        '500000000',
        '500000001',
        '500000002',
        '500000003',
        'Unknown'

    )


@DATASETS.register_module
class Ultra4SimplifiedDataset(XMLDataset):

    CLASSES = (
        '壳牌恒护超凡喜力系列 1L',
        '壳牌恒护超凡喜力系列 4L',
        '壳牌先锋超凡喜力系列 4L',
        '壳牌先锋超凡喜力系列 1L',
        '壳牌超凡喜力系列 1L',
        '壳牌超凡喜力系列 4L',
        '壳牌极净超凡喜力系列 1L',
        '壳牌极净超凡喜力系列 4L',
        '其他'
    )


@DATASETS.register_module
class Ultra4SimplifiedMidDataset(XMLDataset):

    CLASSES = (
        '500000006',
        '500000007',
        '500000005',
        '500000004',
        '500000000',
        '500000001',
        '500000002',
        '500000003',
        'Unknown'
    )




