import time
import argparse
from glob import glob
import shutil,os
 
from sklearn.model_selection import train_test_split
 
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
args = parser.parse_args()

# 去除名字的特殊符号，统一序号视频文件命名
 
# def original_video_name_format():
#     base_path = "../LSR2/main"
#     result = list(glob("{}/*".format(base_path),recursive=False))
#     file_num = 0
#     result_list = []
 
#     for each in result:
#         file_num +=1
#         new_position ="{0}{1}".format( int(time.time()),file_num)
#         result_list.append(new_position)
#         shutil.move(each, os.path.join(base_path,new_position+".mp4"))
#         pass

def trained_data_name_format():
    base_path = args.data_root  # "../LSR2/lrs2_preprocessed_288x288"
    # result = list(glob("{}/*".format(base_path)))
    result = os.listdir(base_path)
    # print(result)
    result_list = []
    for i,dirpath in enumerate(result):
        # shutil.move(dirpath,"{0}/{1}".format(base_path,i))
        # result_list.append(str(i))
        # print('dirpath:', dirpath)
        result_list.append(dirpath)
    if len(result_list)<14:
        test_result=val_result=train_result=result_list
    else:
        train_result,test_result = train_test_split(result_list,test_size=0.15, random_state=42)
        test_result, val_result = train_test_split(test_result, test_size=0.5, random_state=42)
 
    for file_name,dataset in zip(("train.txt","test.txt","val.txt"),(train_result,test_result,val_result)):
        with open(os.path.join("filelists",file_name),'w',encoding='utf-8') as fi:
            for dataset_i in dataset:
                #print('dataset_i:', dataset_i)
                video_result = os.listdir(os.path.join(base_path, dataset_i))
                #print('video_result:', video_result)
                video_result = [dataset_i+'/'+video for video in video_result]
                fi.write("\n".join(video_result))
                fi.write("\n")
 
    # print("\n".join(result_list))

trained_data_name_format()