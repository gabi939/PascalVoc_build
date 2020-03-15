"""

Tool used for annotation process on videos.
Manipulates all the outputs received from cvat annotation tool to create a PascalVoc dataset.

Input received:
1) Video .mp4
2) xml file for each frame from the video


The process:
1) Take video and choose every 60 frame and save it (number of frames can be changed)
2) Take xml files and change them all to much the new frames


"""

import cv2
import os
import xml.etree.ElementTree as ET


def main():
    """
    #process start#

    Here you give the path for the video and the dir containing the annotations.
    Here you choose the amount of frames to skip.

    """
    PATH_VIDEO = r"C:\Users\gabi9\Desktop\Vienna\Chasie_session_2_trial_3.mp4"
    PATH_XML_DIR = r"C:\Users\gabi9\Desktop\temp2"
    frames_skip = 60

    xml_filter(PATH_XML_DIR, frames_skip)
    video_to_frame(PATH_VIDEO, PATH_XML_DIR, frames_skip)


def video_to_frame(video_path, path_xml_dir, frame_skip):
    """
    Frames are saved according to the parameters given
    """

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_skip == 0:  # every 60 frames is being saved
            cv2.imwrite(path_xml_dir + "/frame%d.png" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    print("Finished choosing frames")


def xml_filter(xml_path, frame_skip):
    """
    xml files are chosen and fixed to match the frames
    """
    xml_ls = os.listdir(xml_path)
    xml_ls.sort()
    count = 0

    for xml in xml_ls:
        xml = xml_path + '/' + xml
        tree = ET.parse(xml)
        root = tree.getroot()

        os.remove(xml)

        if count % frame_skip == 0:  # choosing every 60 xml
            for child in root:
                if child.tag == "filename":
                    child.text = "frame" + str(count) + ".png"  # this makes the xml match the frame
                elif child.tag == "object":
                    for sub_child in child:
                        if sub_child.tag == "bndbox":  # fixing bounding box coordinates
                            for sub_sub_child in sub_child:
                                sub_sub_child.text = str(int(float(sub_sub_child.text)))
            tree.write(xml_path + "/frame" + str(count) + ".xml")
        count += 1


if __name__ == '__main__':
    main()
