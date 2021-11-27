from rasterio import rasterio
from tensorflow.python.util import compat
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import pprint
from datetime import datetime

DATASET_ROOT_PATH = '../../data/HiltonOfFern_crop_field_training_cloud_free_available_area'
CLIP_VALUE = 1811.6


# Value is equivalent to 10% of a 256x256 image, will probably need to adjust this value if ever use a different image size!
ACCEPTABLE_CLOUD_SHADOW_SIZE = 6611
ACCEPTABLE_CLOUD_MASK_SIZE = 6293
#ACCEPTABLE_CLOUD_MASK_SIZE = 131071 # ALL BUT ONE PIXEL!
MAX_SENSOR_ERROR = 1348.0
MAX_NO_DATA_VALS = 14319
MAX_SEN1_ERRORS = 26214 # equivalent of 20% of values

def build_dataset(include_ratio, small_testing_dataset=False):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("Started building dataset at ", current_time)
    
    # load list of our sentinel 2 files in random order (so when we split data later we get representative distributions rather than images next to each other - seed randomiser so my work is repeatable!) 
    list_ds = tf.data.Dataset.list_files(DATASET_ROOT_PATH + '/*/*/S2/Patches/*.tif', shuffle=True, seed=42)
    

    if small_testing_dataset:
        total_products = tf.data.experimental.cardinality(list_ds).numpy()
        test_size = int(total_products * 0.90)
        list_ds = list_ds.skip(test_size)


    # remove Sentinel 2 images which are cloudy, covered in shadow, have lots of missing data or sensor errors and Sentinel 1 images that have lots of missing data
    # do all filtering in one Frankenstein function so only have to count dataset size once 
    list_ds = list_ds.filter(filter_low_quality_data)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("finished filtering at ", current_time)
    # unfortunately after filtering tensorflow no longer knows how many items are in dataset and can no longer iterate through it due to this. So have to manually count items and set value
    count = list_ds.reduce(0, lambda x, _: x + 1).numpy()
    tf.print("after filtering have " + str(count) + " files")
    list_ds = list_ds.apply(tf.data.experimental.assert_cardinality(count))
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("finished fixing cardinality at ", current_time)
    

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("Have filtered dataset at ", current_time)

    
    # now actually load info from Sentinel files = create 2 datasets, one that includes the VH/VV ratio for Sentinel 1 files and 1 without)
    list_ds = list_ds.map(lambda x: process_path(x, CLIP_VALUE, "clip", 0, include_ratio ), num_parallel_calls=tf.data.AUTOTUNE)
    list_ds.apply(tf.data.experimental.ignore_errors())
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("Have processed dataset at ", current_time)

    
    # split into test, train, validation and verification sets
    train, val, test, ver = split_dataset(list_ds, tf.data.experimental.cardinality(list_ds).numpy())
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    tf.print("Have finished splitting dataset at ", current_time)
    
    return train, val, test, ver
    
def replace_no_data_in_array(sentinel_data, array, replace_value=0):
  
  array[array == sentinel_data.profile['nodata']] = replace_value
  return array


# dataset contains products outside or on the edge of our region of interest. These have no or very little data so filtering out
# any products which don't contain data for less than the percentage of the image defined by ROI_THRESHOLD at the top of this file
def is_outside_roi(rgb, no_data_value):
    
    no_data_vals =  tf.cast(tf.equal(rgb, no_data_value), tf.int32)
    count_no_data_vals = tf.reduce_sum(no_data_vals)
    
    
    if tf.math.greater(count_no_data_vals, MAX_NO_DATA_VALS): 
       return True
    
    return False
 


def has_significant_sensor_errors(rgb, max_expected_value):
    
    error_values =  tf.math.greater(rgb, max_expected_value)
    num_errors = tf.math.reduce_sum(tf.cast(error_values, tf.float32))
    
    if tf.math.greater(num_errors, MAX_SENSOR_ERROR):
        return True
    
    return False


def filter_sentinel1_missing_data(file_path):
    [no_data_value, sen1] = tf.py_function(load_sen1_data, inp=[file_path], Tout=[tf.float32, tf.float32])
     
    no_data_value_positions = tf.equal(sen1, no_data_value)
    no_data_count = tf.reduce_sum(tf.cast(no_data_value_positions, tf.int32))
    
    if no_data_count > MAX_SEN1_ERRORS:
       return False

    return True
    

def get_no_data_value(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
        
    sentinel_data =  rasterio.open(file_path)
    no_data_value = sentinel_data.profile['nodata']
    sentinel_data.close()
    return float(no_data_value)

# not interested in products which have cloud cover or cloud shadows,have very little data as they were outside our region of interest or have too many extreme values that are probably 
# caused by sensor errors
def filter_low_quality_data(file_path, max_expected_value=1348.0):
    
    # load all the data I need at once to reduce inefficient py_function calls
    [no_data_value, rgb, rgb_with_no_data_vals, cloud_mask, cloud_shadow] = tf.py_function(load_sen2_data, inp=[file_path], Tout=[tf.float32, tf.float32,tf.float32,tf.float32,tf.float32])
    
    # get rid of any images that contain pixels that have a high probablity of containing cloud    
    if tf.math.greater(tf.math.count_nonzero(cloud_mask),ACCEPTABLE_CLOUD_MASK_SIZE):
      return False
    
    # get rid of any images that are in the shadow of a cloud
    if tf.math.greater(tf.math.count_nonzero(cloud_shadow),ACCEPTABLE_CLOUD_SHADOW_SIZE):
      return False
    
    if is_outside_roi(rgb_with_no_data_vals, no_data_value):
        return False
    
   
    # check image doesn't have too many values that seem erroneously high that may have been caused by a sensor error
    if has_significant_sensor_errors(rgb, max_expected_value):
        return False
    
    return filter_sentinel1_missing_data(file_path)
    
# combine loading everything related to sentinel 2 files into one function (as otherwise ended up closing and reopening same file lots of times)
def load_sen2_data(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
        
    sentinel_data =  rasterio.open(file_path)
    
    no_data_value = sentinel_data.profile['nodata']
                                                                           
    red_with_no_data_vals = sentinel_data.read(4)
    red = replace_no_data_in_array(sentinel_data, red_with_no_data_vals)
    
    green_with_no_data_vals = sentinel_data.read(3)
    green = replace_no_data_in_array(sentinel_data, green_with_no_data_vals)
    
    blue_with_no_data_vals = sentinel_data.read(4)
    blue = replace_no_data_in_array(sentinel_data, blue_with_no_data_vals)
    
    rgb = tf.convert_to_tensor(np.dstack((red, green, blue)))   
    rgb_with_no_data_vals = tf.convert_to_tensor(np.dstack((red_with_no_data_vals, green_with_no_data_vals, blue_with_no_data_vals)))   
    
    
    med_prob_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(16))
    high_prob_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(17))
    cirrus_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(18))
    
    cloud_mask = tf.convert_to_tensor(np.dstack((med_prob_cloud, high_prob_cloud, cirrus_cloud)))
    
    cloud_shadow = replace_no_data_in_array(sentinel_data, sentinel_data.read(15))

    sentinel_data.close()
    
    return no_data_value, rgb, rgb_with_no_data_vals, cloud_mask, cloud_shadow
    
# have multiple bands with different cloud information (13 - opaque clouds, 14 cirrus clouds, 15 cloud shadow, 16 medium prob cloud,
# 17 high prob cloud, 18 thin cirrus). I just use 17-18 which can identify images with most obvious clouds
def convert_sen2_product_to_cloud_mask_tensor(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
    
    sentinel_data =  rasterio.open(file_path)

    med_prob_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(16))
    high_prob_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(17))
    cirrus_cloud = replace_no_data_in_array(sentinel_data, sentinel_data.read(18))
    
    cloud_mask = np.dstack((med_prob_cloud, high_prob_cloud, cirrus_cloud))  
    #cloud_mask = np.dstack((high_prob_cloud, cirrus_cloud))  
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(cloud_mask)

def convert_sen2_product_to_cloud_shadow_mask_tensor(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
    
    sentinel_data =  rasterio.open(file_path)
    
    cloud_shadow = replace_no_data_in_array(sentinel_data, sentinel_data.read(15))
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(cloud_shadow)

def replace_outliers(data, max_threshold, method, min_threshold=0):
    if np.min(data) < min_threshold or np.max(data) > max_threshold:
        if method == "clip":
            data = np.clip(data, min_threshold, max_threshold)
        elif method == "ffill":
            # replace all values above threshold with nan
            data[data > max_threshold] = np.nan
            # use pandas to forward fill values
            df = pd.DataFrame(data)
            df.fillna(method='ffill', axis=1, inplace=True)
            data = df.values
        elif method == "mean":
            # set values over our threshold to not a number so they don't skew our mean
            data[data > max_threshold] = np.nan
            mean = np.nanmean(data)
            np.nan_to_num(data, nan=mean, copy=False) 
        else:
            raise ValueError("Invalid outlier replacement method " + outlier_replacement_method + " was passed to convert_sen2_product_to_rgb_tensor function")  
            
    return data

def convert_sen2_product_to_rgb_tensor(file_path, outlier_max_threshold=0, outlier_replacement_method="", outlier_min_threshold=0):
    # sometimes call this function with a string when testing things but when loading my full dataset I use map and it gets called with a string tensor
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy())   
    
    sentinel_data =  rasterio.open(file_path)

    red = replace_no_data_in_array(sentinel_data, sentinel_data.read(4))
    
    if outlier_replacement_method:
        red = replace_outliers(red, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    green = replace_no_data_in_array(sentinel_data, sentinel_data.read(3))
    if outlier_replacement_method:
        green = replace_outliers(green, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    blue = replace_no_data_in_array(sentinel_data, sentinel_data.read(2))
    if outlier_replacement_method:
        blue = replace_outliers(blue, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    rgb = np.dstack((red, green, blue))   
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(rgb)

def load_sen1_data(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 

    sentinel_data =  rasterio.open(file_path)
    
    vh = sentinel_data.read(1)
    vv = sentinel_data.read(2)
                                                                                          
    sen1 = tf.convert_to_tensor(np.dstack((vh,vv)))
                                                                                          
    no_data_value = sentinel_data.profile['nodata']
    sentinel_data.close()
                                                                                          
    return float(no_data_value), sen1
                                                                                          
def convert_sen1_product_to_tensor(file_path, include_ratio = False, replace_no_data = True):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 

    sentinel_data =  rasterio.open(file_path)
    
    if replace_no_data:
        vh = replace_no_data_in_array(sentinel_data, sentinel_data.read(1))
        vv = replace_no_data_in_array(sentinel_data, sentinel_data.read(2))
    else:
        vh = sentinel_data.read(1)
        vv = sentinel_data.read(2)
    
    sentinel_data.close()
    
    vv = tf.convert_to_tensor(vv)
    vh = tf.convert_to_tensor(vh)
    if include_ratio:
        ratio = tf.math.divide_no_nan(vh, vv)
        sen1 = tf.experimental.numpy.dstack((vh,vv, ratio))
    else:    
        sen1 = tf.experimental.numpy.dstack((vh,vv)) 
        
    return sen1

def process_path(file_path, outlier_max_threshold=0, outlier_replacement_method="", outlier_min_threshold=0, include_ratio=False):
    
    sen2 = tf.py_function(func=convert_sen2_product_to_rgb_tensor, inp=[file_path, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold], Tout=[tf.float32]) 
    sen2 = tf.reshape(sen2, [256, 256, 3])

    sen1_file_path = tf.strings.regex_replace(file_path, "/S2/", "/S1") # should there be another / after S1!!!
    sen1 = tf.py_function(func=convert_sen1_product_to_tensor, inp=[file_path, include_ratio], Tout=[tf.float32])
    if include_ratio:
        sen1 = tf.reshape(sen1, [256, 256, 3])
    else:
        sen1 = tf.reshape(sen1, [256, 256, 2])
    
    return sen1, sen2

def split_dataset(dataset, dataset_size):
    train_size = int(0.7 * dataset_size)
    val_size = int(0.14 * dataset_size)
    test_size = int(0.14 * dataset_size)
    ver_size = int(0.2 * dataset_size)

    # split version of our dataset that includes Sentinel 1 ratios in train, test, validation and verification
    train_ds = dataset.take(train_size)  # create train dataset with first 70% of data
    remaining = dataset.skip(train_size)      # store remaining 30% in temp dataset
    val_ds = remaining.take(val_size)          # use next 14% of data for validation
    remaining = remaining.skip(val_size)            # store final 16% in temp dataset
    test_ds = remaining.take(test_size)        # use first 14% of that for test dataset
    ver_ds = remaining.skip(test_size)         # take final 2 percent for verification
    
    return train_ds, val_ds, test_ds, ver_ds
    
    
    
