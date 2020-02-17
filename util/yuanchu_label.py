import random

import xml.etree.ElementTree as ET
import cv2
import os


def solve_xml(annotations_dir):
    for r, dirs, files in os.walk(annotations_dir):
        for file in files:
            if not file.endswith('.xml'):
                continue
            xml_name = file.replace(' ', '')
            print("Parsing %s\n" % os.path.join(r, file))
            tree = ET.parse(os.path.join(r, file))
            root = tree.getroot()
            filename = root.find('filename')
            img_name = filename.text
            path = root.find("path")
            if path is not None:
                path.text = "JPEGImages/" + img_name

            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                name = "text"
                obj.find('name').text = name
            # node_folder = SubElement(root, "folder")
            # node_folder.text = "shell"
            tree.write(os.path.join("/home/lichengzhi/mmdetection/data/VOCdevkit/yuanchu/2020.02.12/Annotations", xml_name), encoding='utf-8')


if __name__ == "__main__":
    solve_xml("/data/lichengzhi/mmdetection/data/VOCdevkit/yuanchu/2020.02.12/Annotations")
