import os
import cv2
import xml.etree.ElementTree as ET
import random
from mmdet.datasets import shell, rosegold, UltraAB

img_source_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/标注图片"
img_output_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/JPEGImages"
xml_source_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.04.23/Annotations"
xml_output_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.04.23/Annotations_alter"





# tags = shell.ShellDataset.CLASSES
# tags = rosegold.RoseGoldDataset.CLASSES
tags = UltraAB.UltraABDataset.CLASSES


def solve_xml():
    print(tags)
    for r, dirs, files in os.walk(xml_source_dir):
        for file in files:
            if not file.endswith('.xml'):
                continue
            xml_name = file.replace(' ', '')
            # pos = xml_name.rfind('.')
            # pos = xml_name[: pos].rfind('.')
            # xml_name = xml_name[: pos] + ".xml"
            print("Parsing %s\n" % os.path.join(r, file))
            tree = ET.parse(os.path.join(r, file))
            root = tree.getroot()

            filename = root.find('filename')
            img_name = filename.text
            img_name = img_name.replace(' ', '')
            pos = img_name.rfind('.')
            img_name = img_name[: pos] + ".jpg"
            filename.text = img_name
            print("xml_name: %s, img_name: %s\n" % (xml_name, img_name))
            assert(xml_name[:-4] == img_name[:-4])

            for obj in root.findall('object'):
                name = obj.find('name').text.strip("\ufeff")
                if name.find("恒护") == -1:
                    name = "其他"
                obj.find('name').text = name
                if name not in tags:
                    print("Label %s in image %s no found.\n" % (name, xml_name))
                assert (name in tags)
            # node_folder = SubElement(root, "folder")
            # node_folder.text = "shell"
            tree.write(os.path.join(xml_output_dir, xml_name), encoding='utf-8')


def solve_img():
    for r, dirs, files in os.walk(img_source_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                img_name = file.replace(' ', '')
                if not img_name.endswith('.jpg'):
                    pos = img_name.rfind('.')
                    img_name = img_name[: pos] + ".jpg"
                cv2.imwrite(os.path.join(img_output_dir, img_name), img)
                if not img_name.endswith('.jpg'):
                    print(img_name)


def generate_main():
    main_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.04.23/ImageSets/Main"
    annotations_dir = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.04.23/Annotations"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    total_xml = os.listdir(annotations_dir)
    num = len(total_xml)
    filelist = range(num)
    ftrainval = open(os.path.join(main_dir, "trainval.txt"), "w")
    ftest = open(os.path.join(main_dir, "test.txt"), "w")
    ftrain = open(os.path.join(main_dir, "train.txt"), "w")
    fval = open(os.path.join(main_dir, "val.txt"), "w")

    for i in filelist:
        name = total_xml[i][:-4] + '\n'
        x = random.random()
        if x < 0.05:
            ftest.write(name)
        else:
            ftrainval.write(name)
            if 0.05 <= x < 0.1:
                fval.write(name)
            else:
                ftrain.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def main():
    # solve_img()
    # solve_xml()
    generate_main()


if __name__ == "__main__":
    main()
