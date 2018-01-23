import numpy as np
from numpy import array
import tensorflow as tf
import pickle
import sys
import random
import PIL
from PIL import Image
import csv
import pandas as pd
import os
#from tensorflow.python.platform import gfile
def process_gender(list_of_genders):
    #we assume the values to be 'M, 'F', others
    ret_list=[]
    for x in list_of_genders:
        if x == 'M':
            ret_list.append(0)
        elif x == 'F':
            ret_list.append(1)
        else:
            ret_list.append(2)
    return np.array(ret_list)

def process_class_names(list_of_class_names, num_classes):
    # we assume the class names to be class_i where i is between 1 and 14, and num_classes is 15 (one for other?)
    ret_list=[]
    index = len("class_")
    for x in list_of_class_names:
        ret_list.append(int(x[index:]))
    class_array=np.array(ret_list).astype(np.int)
    #print(str(class_array.tolist()))
    one_hot_array= (np.arange(num_classes) == class_array[:,None]-1).astype(int)
    return one_hot_array

def load_csv_file_without_images( csv_file, num_classes, debug_logs):
    input = pd.read_csv(csv_file, sep=',', header=0)
    #row_id,age,gender,view_position,image_name,detected
    ages = input.iloc[:,1].values
    #print(ages.shape)
    #print(ages.dtype)
    #print(ages.tolist())
    gender = input.iloc[:,2].values
    #print(gender.tolist())
    gender_numeric = process_gender(gender.tolist())
    #print(gender_numeric.tolist())
    view_position = input.iloc[:,3].values
    #print(view_position.tolist())
    image_names = input.iloc[:,4].values
    first = True
    #Create other meta features
    meta_np = np.expand_dims(ages, axis=1)
    gender_numeric_exp = np.expand_dims(gender_numeric, axis=1)
    meta_np = np.append(meta_np, gender_numeric_exp, axis=1)
    view_position_exp = np.expand_dims(view_position, axis=1)
    meta_np = np.append(meta_np, view_position_exp, axis=1)
    images_names_exp = np.expand_dims(image_names, axis=1)
    meta_np = np.append(meta_np, images_names_exp, axis=1)
    #print(meta_np.tolist())

    #create output class labels
    class_names = input.iloc[:,5].values
    #print(class_names.tolist())
    class_names_numeric = process_class_names(class_names.tolist(), num_classes)
    #print(class_names_numeric.tolist())
    return  meta_np, class_names_numeric
    #WriteLog(debug_logs,str(image_array.tolist()))
    #print(str(input.iloc[2,2]))

def load_csv_file(img_dir_name, csv_file,num_classes, debug_logs):
    input = pd.read_csv(csv_file, sep=',', header=0)
    #row_id,age,gender,view_position,image_name,detected
    ages = input.iloc[:,1].values
    #print(ages.shape)
    #print(ages.dtype)
    #print(ages.tolist())
    gender = input.iloc[:,2].values
    #print(gender.tolist())
    gender_numeric = process_gender(gender.tolist())
    #print(gender_numeric.tolist())
    view_position = input.iloc[:,3].values
    #print(view_position.tolist())
    image_names = input.iloc[:,4].values.tolist()
    first = True
    for x in image_names:
        try:
            image_np = CreateVector(img_dir_name, x, debug_logs)
            if(first):
                image_array = np.expand_dims(image_np, axis=0)
                first = False
            else:
                new_image_array = np.expand_dims(image_np, axis=0)
                image_array = np.append(image_array, new_image_array, axis=0)
                #print(image_array.shape)
        except Exception:
            print("File not found")
    #Create other meta features
    meta_np = np.expand_dims(ages, axis=1)
    gender_numeric_exp = np.expand_dims(gender_numeric, axis=1)
    meta_np = np.append(meta_np, gender_numeric_exp, axis=1)
    view_position_exp = np.expand_dims(view_position, axis=1)
    meta_np = np.append(meta_np, view_position_exp, axis=1)
    #print(meta_np.tolist())

    #create output class labels
    class_names = input.iloc[:,5].values
    #print(class_names.tolist())
    class_names_numeric = process_class_names(class_names.tolist(), num_classes)
    #print(class_names_numeric.tolist())
    return image_array, meta_np, class_names_numeric
    #WriteLog(debug_logs,str(image_array.tolist()))
    #print(str(input.iloc[2,2]))

def CreateVector(img_dir_name, image_file):

    #print( "Inside Create vector")
    #Read Image
    try:
        file_path= img_dir_name+str(os.sep)+image_file
        #print("file path is "+file_path)
        img = PIL.Image.open(file_path).convert("L") 
    # Need to ignore the RGB images
    except FileNotFoundError:
        print("File not found "+ image_file)
        #raise Exception
    train_x = np.array(img.getdata())
    #recreateImage(img.mode, img.size, train_x, "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\generatedImages\\"+image_file)
    #WriteLog(debug_logs,str(train_x.tolist()))
    #print("size of image data"+str(train_x.shape))
    return train_x.astype(float)

def recreateImage(mode, size, data, file_name):
    img = Image.new(mode, size)
    img.putdata(data.tolist())
    img.save(file_name)
    #im2.save("C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\generatedImages\\"+image_file)
def WriteLog(debug_logs, entry):
    debug_logs.write(entry)
    debug_logs.write("\n")


if __name__ == "__main__":
    args = sys.argv
    img_dir_name = args[1]
    debug_log_file = args[3]
    csv_file = args[2]
    num_classes=14
    debug_logs = open(debug_log_file, 'w')
    np.set_printoptions(threshold='nan')
    load_csv_file(img_dir_name, csv_file,num_classes, debug_logs)
    debug_logs.close()
    #CreateVector(img_dir_name, csv_file, debug_log_file)
    #print("Enter any char...")
    #name = sys.stdin.readline()
    #"C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\sampleImages" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\train_sample2.csv" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\log1.txt"