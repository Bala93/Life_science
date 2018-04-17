import glob
import os
import cv2
import shutil as sh
import random


def frameCreate(vid_path,frame_save_path):

    # Create frame start with 0
    # print vid_path
    cap  = cv2.VideoCapture(vid_path)
    frame_count = 0
    success,frame = cap.read()
    
    while success:
        filename = os.path.join(frame_save_path,'%.4d.jpg'%(frame_count))
        cv2.imwrite(filename,frame)
        success,frame = cap.read()
        frame_count += 1

    return



if __name__ == "__main__":
    
    ###----------------------------------------------------------------#######  
    # Place the text files generated from the DarkNet tool in a single folder# 
    # Specify the output path #
    # Mention the dimension of the image #
    # 
    ###----------------------------------------------------------------#######
    

    ## Get this variables from command line.

    path = '/media/balamurali/New Volume2/IIT-HTIC/GE_Project/Train/'
    out_path = '/media/balamurali/New Volume2/IIT-HTIC/GE_Project/Dataset/'

    
    ## Training test split
    split = 0.8
    clear_folder = True
    dim = 540.0
    ## Get list of txt files
    gt_file_format  = 'txt'
    gt_file_path = []
    ## Get list of video files 
    vid_file_format = 'avi'
    vid_file_path = []
    # if clear_folder:
        

    # Get list of text files ground truth
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".%s"%(gt_file_format))]:
            file_path = os.path.join(dirpath,filename)
            gt_file_path.append(file_path)
    
    print "No of Ground truth files found : {}".format(len(gt_file_path))
    
    
    
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".%s"%(vid_file_format))]:
            file_path = os.path.join(dirpath,filename)
            vid_file_path.append(file_path)

    print "No of Video files found : {}".format(len(vid_file_path))
    ## Create frames

    for f_count,vid in enumerate(vid_file_path):
        vid_dir_path    = '/'.join(vid.split('/')[:-1])
        frame_save_path = os.path.join(vid_dir_path,'Images')
        print "Creating frames for {}".format(vid)    
        if not os.path.exists(frame_save_path):
            os.mkdir(frame_save_path)
        
        frameCreate(vid,frame_save_path)
        


    gt_save_path = os.path.join(out_path,'GT')
    # This is to know what is the order of the text file considered.
    gt_order_path = os.path.join(out_path,'GT_order.txt') 
    
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)

    ## Each video gives a single text file. 
    ## Get all such text files
    with open(gt_order_path,'w') as f:
        for ii in (gt_file_path):
            f.write(ii + '\n')
    
    for file_count,txt in enumerate(gt_file_path):
        print "Extracting bounding box from {}".format(txt)
        with open(txt,'r') as f1:
            lines = f1.readlines()
            for each in lines:
                info = each.strip().split(',')
                general_info = [int(i) for i in info] 
                frame_no  = '%.4d'%(general_info[0])
                no_bounding_box = general_info[1]
                bounding_box_info = general_info[2:]

                for each_box in range(no_bounding_box):
                    # No 5 denotes the format cx,cy,w,h,label
                    start = each_box * 5
                    end   = (each_box + 1) * 5
                    each_bounding_box_info = bounding_box_info[start:end]
                    gt_out_file_name = "%d_%s.txt"%(file_count,frame_no)
                    out_gt_save_file = os.path.join(gt_save_path,gt_out_file_name)
                
                    with open(out_gt_save_file,'a') as f2:                          
                        cx = each_bounding_box_info[0] / dim
                        cy = each_bounding_box_info[1] / dim
                        w  = each_bounding_box_info[2] / dim
                        h  = each_bounding_box_info[3] / dim
                        class_type = each_bounding_box_info[4]
                        coord_string = '%d %f %f %f %f\n'%(class_type,cx,cy,w,h)                
                        f2.write(coord_string)


    ## Making image folder with respect to text file.
    print "Creating images with respect to the annotated text files"
    img_out_path = os.path.join(out_path,'Images')
    if not os.path.exists(img_out_path):
        os.mkdir(img_out_path)

    gt_txt_files = glob.glob(gt_save_path + '/*.txt')
    for ii in gt_txt_files:
        text_file_name = ii.split('/')[-1][:-4]
        folder_no = text_file_name.split('_')[0]
        file_no   = text_file_name.split('_')[1]
        images_path = os.path.join('/'.join(vid_file_path[int(folder_no)].split('/')[:-1]),'Images')
        src_image_name  = '%s.jpg'%(file_no)
        src = os.path.join(images_path,src_image_name)
        dst_image_name  = '%s_%s.jpg'%(folder_no,file_no)
        dst = os.path.join(img_out_path,dst_image_name)
        # print src,dst
        sh.copy(src,dst)
        # break
    
    gt_save_path = os.path.join(out_path,'GT')
    gt_txt_files = glob.glob(gt_save_path + '/*.txt')

    train_path = os.path.join(out_path,'Train')
    test_path  = os.path.join(out_path,'Test') 

    if not os.path.exists(train_path):
        os.mkdir(train_path)

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    train_img_path = os.path.join(train_path,'Images')
    train_gt_path  = os.path.join(train_path,'GT')
    test_img_path  = os.path.join(test_path,'Images')
    test_gt_path   = os.path.join(test_path,'GT')

    if not os.path.exists(train_img_path):
        os.mkdir(train_img_path)

    if not os.path.exists(train_gt_path):
        os.mkdir(train_gt_path)

    if not os.path.exists(test_img_path):
        os.mkdir(test_img_path)

    if not os.path.exists(test_gt_path):
        os.mkdir(test_gt_path)

    random.shuffle(gt_txt_files)
    total_len = len(gt_txt_files)
    train_len = int(split * total_len)
    test_len  = total_len - train_len

    print "Splitting data for Train and Test"
    print "Total Data : {}".format(total_len)
    print "Train Data : {}".format(train_len)
    print "Test Data  : {}".format(test_len)

    for count,ii in enumerate(gt_txt_files):
        file_path = ii
        file_name = file_path.split('/')[-1][:-4]
        if count < train_len:
            src_text = os.path.join(out_path,'GT',file_name + '.txt')
            dst_text = os.path.join(train_gt_path,file_name + '.txt')
            src_img  = os.path.join(out_path,'Images',file_name + '.jpg')
            dst_img  = os.path.join(train_img_path,file_name + '.jpg')
        else:
            src_text = os.path.join(out_path,'GT',file_name + '.txt')
            dst_text = os.path.join(test_gt_path,file_name + '.txt')
            src_img  = os.path.join(out_path,'Images',file_name + '.jpg')
            dst_img  = os.path.join(test_img_path,file_name + '.jpg')

        sh.copy(src_text,dst_text)
        sh.copy(src_img,dst_img)