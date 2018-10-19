'''
要求：
1）文件夹根目录
2）每个图片与对应标定xml文件放同级目录
3）clsasses与对应标定文件一致
4)标定图片为jpg格式，其他格式需要修改代码
功能：
1)路径下所有.xml文件自动处理；
2)xml转txt:txt放在与xml相同路径,有 目标.txt、train.txt(内部存放绝对路径)
3)统计每一类别的anchor的宽高比
遍历记录所有xml里的宽高比，计算出所有的宽高比，取一个中值或均值，每一个类别输出一个比例
4）各类别的数量
5）各类别占原图像比
遍历记录所有xml里的宽占原图像的比例，计算出所有的宽高比，取一个中值或均值，每一个类别输出一个比例
'''
import argparse
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir
from os.path import join
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
import copy

classes = ["upbody", "bicycle", "motor", "tricycle", "dog", "car"]


def convert_annotation(image_name):
    in_file = open('%s.xml'%(image_name))
    #out_file = open('%s/%s.txt'%(path, image_name), 'w')
    tree=ET.parse(in_file) #解析xml文件
    root = tree.getroot()
    size = root.find('size')
    w = float(size.find('width').text)#1920
    h = float(size.find('height').text)#1080

    ls = list()
    ls1 = list()
    lss =[]
    ls1s =[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)#通过查表获得对应指数
        
        xmlbox = obj.find('bndbox')
        
        xmin = float(xmlbox.find('xmin').text)
        ymin = float(xmlbox.find('ymin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymax = float(xmlbox.find('ymax').text)
        #kuan=xmax-xmin
        #gao=ymax-ymin
        #宽高比
        aspectratio= (xmax-xmin)/(ymax-ymin)
        #类别占原图像比
        widthratio= (xmax-xmin)/w
        #print(cls_id, aspectratio)
        #new_list=[aspectratio,widthratio]
        
        ls.append([cls_id,aspectratio,widthratio])
            #ls.append(new_list)
        #lss.append(ls)
    return ls
    
def getListFiles(path):
    paths = [] 
    for root, dirs, files in os.walk(path):  
        for dir_ in dirs: 
            paths.append(os.path.join(root,dir_)) 
    paths.append(path) #需要将根目录也算进去
    return paths

def arg_parse():
    """
    Parse arguements to the sample process module
    
    """
    
    parser = argparse.ArgumentParser(description="PASCALxml To YOLOtxt && RoiCrop")
    parser.add_argument("--srcpath", dest = "srcpath", help ="srcpath",default = "null", type = str)
    
    return parser.parse_args()

if __name__=="__main__":
    args = arg_parse()
    src_path = args.srcpath
    
    #遍历出所有文件夹
    paths = getListFiles(src_path)
    print(paths)
    #print("**********\n")
    #print(len(paths))
    li00 = []
    li01 = []
    li02 = []
    li03 = []
    li04 = []


    for path in paths: 
        #print("%s processing..." %(path))
        

        
        xml_paths = glob.glob(path+'/*.xml')#获取路径下所有xml文件
        
        li02 = []
        for xml_path in xml_paths:
            #print(xml_paths)
            #print('---------------------')
            image_name = os.path.splitext(xml_path.split('/')[-1])[0]
            #list_file.write('%s/%s.jpg\n'%(path, image_name)) #把图片名写入train.txt
            li00= convert_annotation(image_name)
            li02.extend(li00)

        li01.extend(li02)
    list_file = open('kuangaobi.txt', 'w')
    list1_file = open('kuanzhanyuantubi.txt', 'w')
    # for i in range(len(li01)):
        # for 
    data = []
    for i in range(len(classes)):
        print(i)
        count =0
        ret = []
        for j in range(len(li01)):

            if (li01[j][0] == i):
                if(count==0):
                    print("{}".format(i),file=list1_file)
                    ret.append(i)
                    print(i)
                ret.append(li01[j][1:])
                
                #print(li01[j],file=list_file)
                #print(len(li01[j]))
                
                #print(li01[j],)
                count+=1
        data.append(ret)
        print(ret,file=list1_file)
        
        print("{}类共有{}".format(i,count),file=list_file)
        
    
    print('-----------------')
    kmeanss = []
    kmeans = KMeans(n_clusters=3)
    for i in range(len(data)):
        if len(data[i]) != 0:
            if data[i][0]>=0 and data[i][0]<=len(classes):
                y = np.array(data[i][1:])
                
                kmeans.fit(y)
                centroids = kmeans.cluster_centers_ 
                print("success")
                print("{}的聚类中心是\n{}".format(i,centroids),file=list_file)
    # 
    # 
    # 
    # print (centroids)
    # print (centroids,file = list_file)   

    # data_1=[[i] for i in li04]
    # kmeans_1 = KMeans(n_clusters=3)
    # kmeans_1.fit(data_1)
    # centroids_1 = kmeans_1.cluster_centers_ 
    # print (centroids_1)
    # print (centroids_1,file = list1_file)

    #list_file.close()

#修改成将所有train.txt文件合并成一个总的train.txt
#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt") #调用终端命令进行文件合并
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

