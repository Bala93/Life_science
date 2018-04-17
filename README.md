
This repository is to keep track of the codes carried out in GE Project

yolo_img_txt - It is to get the the images and the respective ground truth files in a format which is expected by yolo

Yolo expects the file structure of this following way:

--train_data
---train_images
----*.jpg
---train_labels
----*.txt
--test_data
---test_images
----*.jpg
---test_labels
----*.txt

Darlabel annotation tool is used to create annotation dataset. The tool gives a .txt file with all the maked bounding box.

The code yolo_img_txt takes this .txt file and gives the folder with images and txt files separately



