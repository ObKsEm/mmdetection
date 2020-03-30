import os
import random
import shutil
import sys

xml_source_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2019.11.19/Annotations"
img_source_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2019.11.19/JPEGImages"

xml_target_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.03.25/Annotations_rg"
img_target_path = "/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.03.25/JPEGImages_rg"


def main():
    if os.path.exists(xml_target_path):
        os.removedirs(xml_target_path)
    if os.path.exists(img_target_path):
        os.removedirs(img_target_path)
    os.makedirs(img_target_path)
    os.makedirs(xml_target_path)
    for r, dirs, files in os.walk(xml_source_path):
        for file in files:
            x = random.random()
            if x < 1.0 / 6.0:
                print(file)
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