import csv
import os
from collections import defaultdict

image_dir = '/home/lichengzhi/mmdetection/data/VOCdevkit/SKU110K/JPEGImages'
xml_dir = '/home/lichengzhi/mmdetection/data/VOCdevkit/SKU110K/Annotations'
main_dir = '/home/lichengzhi/mmdetection/data/VOCdevkit/SKU110K/ImageSets/Main'


def save_xml(image_name, entry):
    from lxml.etree import Element, SubElement, tostring
    from xml.dom.minidom import parseString

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = "%s" % entry.get('width')

    node_height = SubElement(node_size, 'height')
    node_height.text = "%s" % entry.get('height')

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = "3"

    for item in entry["items"]:
        xmin, ymin, xmax, ymax = item["bbox"]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = "%s" % item["class"]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = "0"
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = "%s" % xmin
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = "%s" % ymin
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = "%s" % xmax
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = "%s" % ymax

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    output_xml = os.path.join(xml_dir, image_name.replace('jpg', 'xml'))
    with open(output_xml, 'wb') as f:
        f.write(xml)

    return


def convert_to_xml(label_dict):
    for image_name in label_dict.keys():
        entry = label_dict[image_name]
        save_xml(image_name, entry)


def read_csv(csv_path, pre_dir):
    ret = defaultdict(dict)
    for csv_filename in ['annotations_test.csv', 'annotations_train.csv', 'annotations_val.csv']:
        with open(os.path.join(csv_path, csv_filename), "r") as f:
            reader = csv.reader(f)
            header = True
            for line in reader:
                # 除去文件头
                if header:
                    header = False
                    continue
                # 处理文件存储路径，当做标签
                name = line[0]
                image_path = os.path.join(pre_dir, name)
                # 处理后面的bbox
                bbox = line[1:5]
                c = line[5]
                width = int(line[6])
                height = int(line[7])
                entry = ret[name]
                entry["path"] = image_path
                entry["width"] = width
                entry["height"] = height
                if entry.get("items", None) is None:
                    entry["items"] = list()
                entry["items"].append({
                    "bbox": bbox,
                    "class": c
                })

    return ret


def generate_main(label_dict):
    ftrainval = open(os.path.join(main_dir, "trainval.txt"), "w")
    ftest = open(os.path.join(main_dir, "test.txt"), "w")
    ftrain = open(os.path.join(main_dir, "train.txt"), "w")
    fval = open(os.path.join(main_dir, "val.txt"), "w")
    
    for image_name in label_dict.keys():
        name = image_name[: -4] + '\n'
        if name.find("test") > -1:
            ftest.write(name)
        elif name.find("train") > -1:
            ftrain.write(name)
            ftrainval.write(name)
        elif name.find("val") > -1:
            fval.write(name)
            ftrainval.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    csv_path = '/home/lichengzhi/mmdetection/data/SKU110K/annotations'
    label_dict = read_csv(csv_path=csv_path, pre_dir=image_dir)
    convert_to_xml(label_dict)
    generate_main(label_dict)

