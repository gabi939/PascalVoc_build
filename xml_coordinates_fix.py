"""




This file is used for fixing PascalVoc annotation file.
The fix is changing the float numbers into ints in the coordinates of the bounding boxes.



"""

__author__ = "Gabi Malin"


import os
import xml.etree.ElementTree as ET


def main():
    """
    get all xml files and start fixing process
    """

    PATH = "/home/gabi/Desktop/temp2"

    ls_xml = os.listdir(PATH)

    for xml in ls_xml:
        xml = PATH + "/" + xml
        fix_xml(xml)


def fix_xml(xml):
    """
    change xml file coordinates to ints
    :param xml:
    :return:
    """
    tree = ET.parse(xml)
    root = tree.getroot()

    for child in root:
        if child.tag == "object":
            for sub_child in child:
                if sub_child.tag == "bndbox":
                    for sub_sub_child in sub_child:
                        sub_sub_child.text = str(int(float(sub_sub_child.text)))

    tree.write(xml)


if __name__ == '__main__':
    main()
