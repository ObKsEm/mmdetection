import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import operator as op


xml_source_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/Annotations_ori"
img_source_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17/JPEGImages_ori"

xml_target_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.05.29/Annotations"
img_target_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.05.29/JPEGImages"


def merge_conditional(xml_path):
    # tree = ET.parse(xml_path)
    # root = tree.getroot()
    # objs = root.findall('object')
    # for ix, obj in enumerate(objs):
    #     name = obj.find('name').text
    #     if op.eq(name, "壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 1L") or op.eq(name, "壳牌先锋超凡喜力 ACEA C5 天然气全合成油 0W-20 4L"):
    #         return True
    x = random.random()
    if x < 1.0 / 8.0:
        return True
    else:
        return False


def main():
    if os.path.exists(xml_target_path):
        os.removedirs(xml_target_path)
    if os.path.exists(img_target_path):
        os.removedirs(img_target_path)
    os.makedirs(img_target_path)
    os.makedirs(xml_target_path)
    for r, dirs, files in os.walk(xml_source_path):
        for file in files:
            print(file)
            if merge_conditional(os.path.join(r, file)):
                xml_name = file
                img_name = file[:-4] + ".jpg"
                try:
                    shutil.copy(os.path.join(xml_source_path, xml_name), os.path.join(xml_target_path, xml_name))
                    shutil.copy(os.path.join(img_source_path, img_name), os.path.join(img_target_path, img_name))
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())


if __name__ == "__main__":
    main()
