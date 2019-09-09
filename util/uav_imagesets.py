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
                name = name.replace('_', ' / ')
                obj.find('name').text = name
                assert name == "uav"
            # node_folder = SubElement(root, "folder")
            # node_folder.text = "shell"
            tree.write(os.path.join(annotations_dir, xml_name), encoding='utf-8')


def main():
    annotations_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/uav/aug/Annotations"
    image_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/uav/aug/JPEGImages"
    output_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/uav/aug/ImageSets/Main"
    filenames = []
    for r, _, files in os.walk(image_dir):
        for file in files:
            img = cv2.imread(os.path.join(image_dir, file))
            if img is not None:
                filenames.append(file[:-4])

    print("Load %d image files." % len(filenames))

    ftrainval = open(os.path.join(output_dir, "trainval.txt"), "w")
    ftest = open(os.path.join(output_dir, "test.txt"), "w")
    ftrain = open(os.path.join(output_dir, "train.txt"), "w")
    fval = open(os.path.join(output_dir, "val.txt"), "w")

    for filename in filenames:
        rd = random.random()
        if 0 <= rd < 0.05:
            ftest.write(filename)
            ftest.write("\n")
        elif 0.05 <= rd < 0.1:
            fval.write(filename)
            fval.write("\n")
        else:
            ftrain.write(filename)
            ftrain.write("\n")
            ftrainval.write(filename)
            ftrainval.write("\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    solve_xml(annotations_dir)


if __name__ == "__main__":
    main()
    # solve_xml("/home/lichengzhi/mmdetection/data/VOCdevkit/uav/aug/Annotations")
