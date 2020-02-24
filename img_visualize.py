"""


This file is used as a debug tool for PascalVoc image-annotation relation,
Just visualizes the results of the annotation on the image.


NOTE: you choose one image and one annotation

"""

import xml.etree.ElementTree as ET
import cv2


def main():
    """
    Put the path of the chosen xml and image and start visualize process
    """

    IMAGE = '/home/gabi/Desktop/data_for_training (copy)/Tel_Mond/frame123.png'
    ANNOTATION = '/home/gabi/Desktop/data_for_training (copy)/Tel_Mond/frame123.xml'

    show(IMAGE, ANNOTATION)
    exit(-1)


def read_content(xml_file: str):
    """
    Gets all the bounding boxes coordinates inside the xml file
    :param xml_file:
    :return: list of bounding boxes
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


def show(image_path, annotation_path):
    """
    draws annotation on the image and displays it
    """
    img = cv2.imread(image_path)
    points_ls = read_content(annotation_path)
    for point in points_ls:
        point_A = (point[0], point[1])
        point_B = (point[2], point[3])
        img = cv2.rectangle(img, point_A, point_B, 200)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
