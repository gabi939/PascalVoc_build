"""

This file takes folder containing different datasets in different folders,
and merges them to one dataset in a single folder.

"""

__author__ = "Gabi Malin"

import os
import xml.etree.ElementTree as ET


def main():
    """
    merge process start
    :return:
    """
    # path of dataset folder and output folder
    DATASETS_DIR = "/home/gabi/Desktop/data_for_training (copy)"
    MERGED_DATASETS_DIR = DATASETS_DIR + "/merged"

    datasets = os.listdir(DATASETS_DIR)
    os.mkdir(MERGED_DATASETS_DIR)

    count = 0

    for dataset in datasets:
        dataset = DATASETS_DIR + "/" + dataset

        xml_buffer = get_xml_files(dataset)
        frame_buffer = get_frames(dataset)

        # rewriting frames and xml files to a new folder with a new name
        for frame, xml in zip(frame_buffer, xml_buffer):
            rewrite_frame_name(dataset + "/" + frame, MERGED_DATASETS_DIR, count)
            rewrite_xml_name(dataset + "/" + xml, MERGED_DATASETS_DIR, count)
            count += 1

        xml_buffer.clear()
        frame_buffer.clear()


def rewrite_xml_name(old_path, new_path, new_num):
    """
    updates the xml to refer to new frame and moves xml to new area with a new name
    """
    new_name = "frame" + str(new_num)
    tree = ET.parse(old_path)
    root = tree.getroot()

    for child in root:
        if child.tag == "filename":
            child.text = new_name + ".png"
            break

    tree.write(old_path)
    os.rename(old_path, new_path + "/" + new_name + ".xml")


def rewrite_frame_name(old_path, new_path, new_num):
    """
    moves frame to new area with a new name
    """
    os.rename(old_path, new_path + "/" + "frame" + str(new_num) + ".png")


def get_xml_files(path):
    """
    gets all xml files from folder sorts them and returns a sorted list of xml files
    """
    ls_old = os.listdir(path)
    ls_new = []

    for index in range(len(ls_old)):
        if ls_old[index].endswith(".xml"):
            ls_new.append(ls_old[index])

    ls_old.clear()
    ls_new.sort(key=lambda filename: sort_condition(filename))
    return ls_new


def get_frames(path):
    """
    gets all frames from folder sorts them and returns a sorted list of frames files
    """
    ls_old = os.listdir(path)
    ls_new = []

    for index in range(len(ls_old)):
        if ls_old[index].endswith(".png"):
            ls_new.append(ls_old[index])

    ls_old.clear()
    ls_new.sort(key=lambda filename: sort_condition(filename))
    return ls_new


def change_filename_inside_xml():
    pass


def sort_condition(filename):
    """
    returns the number of the file

    example:
    frame1.xml -> 1
    frame200.png -> 200
    """
    filename = filename[5:]
    filename = filename[:-4]
    return int(filename)


if __name__ == '__main__':
    main()
