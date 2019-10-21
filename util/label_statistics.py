import os
import xml.etree.ElementTree as ET
from mmdet.datasets.shell import ShellDataset
from mmdet.datasets.rosegold import RoseGoldDataset


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        coords.append(name)
    return coords


def main():
    xml_base_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2019.10.16/Annotations"
    # xml_base_path = "/home/lichengzhi/CV_ToolBox/data/2019.8.1/Annotations"
    classes = RoseGoldDataset.CLASSES
    d = dict()
    for label in classes:
        d[label] = 0.0
    sum = 0
    for r, dirs, files in os.walk(xml_base_path):
        for file in files:
            print(file)
            xml_file = os.path.join(r, file)
            coords = parse_xml(xml_file)
            for label in coords:
                sum += 1.0
                d[label] += 1
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    with open("data_distribution.txt", "w") as f:
        for k, v in d:
            f.write("%s %d %.6f\n" % (k, v, v / sum))


if __name__ == "__main__":
    main()
