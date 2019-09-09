from .voc import VOCDataset


class UavDataset(VOCDataset):

    CLASSES = ('uav', )
