from rasterio import rasterio
from tensorflow.python.util import compat
import numpy as np
import pandas as pd
import tensorflow as tf
import sys



# Value is equivalent to 10% of a 256x256 image, will probably need to adjust this value if ever use a different image size!
ACCEPTABLE_CLOUD_SHADOW_SIZE = 6554
SENSOR_ERROR_THRESHOLD = 0.125

def replace_no_data_in_array(sentinel_data, array, replace_value=0):
  
  amount_no_data = (array == sentinel_data.profile['nodata']).sum()
  print("Found " + str(amount_no_data) + " no data values in array")
  array[array == sentinel_data.profile['nodata']] = replace_value
  amount_no_data = (array == sentinel_data.profile['nodata']).sum()
  print("Found " + str(amount_no_data) + " no data values in array after replacing them")  
  return array


# dataset contains products outside or on the edge of our region of interest. These have no or very little data so filtering out
# any products which don't contain data for less than the percentage of the image defined by ROI_THRESHOLD at the top of this file
def is_outside_roi(rgb_values):
    non_zero = tf.experimental.numpy.nonzero(rgb)
    total_values = tf.cast(tf.size(rgb), dtype=tf.float32)
    
    threshold_percentage = tf.constant(ROI_THRESHOLD, dtype=tf.float32)
    threshold = tf.math.multiply(total_values, threshold_percentage)
    
    if tf.math.less(tf.cast(tf.size(non_zero), dtype=tf.float32), threshold):
       return True
    
    return False
 
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
    
    sentinel_data =  rasterio.open(file_path)
    no_data_count = tf.reduce_sum(tf.cast(tf.equal(rgb_values, sentinel_data.profile['nodata']), tf.int32))
    sentinel_data.close()
    
    if tf.math.greater(no_data_count, 0):
        return True
    return False

def has_significant_cloud_shadow(cloud_shadow_mask):
    total_mask_values = tf.size(cloud_shadow_mask)
    
    if tf.math.greater(total_mask_values, ACCEPTABLE_CLOUD_SHADOW_SIZE):
       return True
    return False

def has_significant_sensor_errors(rgb, max_expected_value):
    error_values =  tf.math.greater(rgb, max_expected_value)
    num_errors = tf.math.reduce_sum(tf.cast(error_values, tf.float32))

    error_threshold_percentage = tf.constant(0.125, dtype=tf.float32)
    error_threshold = tf.math.multiply(total_values, error_threshold_percentage)
    
    if tf.math.greater(num_errors, error_threshold):
        return True
    
    return False

# not interested in products which have cloud cover or cloud shadows,have very little data as they were outside our region of interest or have too many extreme values that are probably 
# caused by sensor errors
def filter_low_quality_images(file_path, max_expected_value=1348.0):
   
    # get rid of any images that contain pixels that have a high probablity of containing clouds
    cloud_mask = tf.py_function(convert_sen2_product_to_cloud_mask_tensor, inp=[file_path], Tout=[tf.float32])
    
    if tf.math.greater(tf.math.count_nonzero(cloud_mask),0):
      return False
    
    # get rid of any images that are in the shadow of a cloud
    cloud_shadow_mask = tf.py_function(convert_sen2_product_to_cloud_shadow_mask_tensor, inp=[file_path], Tout=[tf.float32])
    
    if has_significant_cloud_shadow(cloud_shadow_mask):
       return False
    
    rgb = tf.py_function(convert_sen2_product_to_rgb_tensor, inp=[file_path], Tout=[tf.float32])
    rgb = tf.reshape(rgb, [256, 256, 3])
    
    if is_outside_roi(rgb):
        return False

    # check image doesn't have too many values that seem erroneously high that may have been caused by a sensor error
    if has_significant_sensor_errors(rgb, max_expected_value):
        return False
    
    return True

# have multiple bands with different cloud information (13 - opaque clouds, 14 cirrus clouds, 15 cloud shadow, 16 medium prob cloud,
# 17 high prob cloud, 18 thin cirrus). I just use 17-18 which can identify images with most obvious clouds
def convert_sen2_product_to_cloud_mask_tensor(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
    #tf.print(file_path)    
    sentinel_data =  rasterio.open(file_path)

    med_prob_cloud = sentinel_data.read(16)
    high_prob_cloud = sentinel_data.read(17)
    cirrus_cloud = sentinel_data.read(18)
    
    cloud_mask = np.dstack((med_prob_cloud, high_prob_cloud, cirrus_cloud))  
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(cloud_mask)

def convert_sen2_product_to_cloud_shadow_mask_tensor(file_path):
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy()) 
    
    sentinel_data =  rasterio.open(file_path)
    
    cloud_shadow = sentinel_data.read(15)
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(cloud_shadow)

def replace_outliers(data, max_threshold, method, min_threshold=0):
    print("in  replace_outliers min threshold = " + str(min_threshold))
    if np.min(data) < min_threshold or np.max(data) > max_threshold:
        if method == "clip":
            print("min value before clipping is " + str(np.min(data)))
            data = np.clip(data, min_threshold, max_threshold)
            print("min value after clipping is " + str(np.min(data)))
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

def convert_sen2_product_to_rgb_tensor(file_path, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold=0):
    print("in  convert_sen2_product_to_rgb_tensor min threshold = " + str(outlier_min_threshold))
    # sometimes call this function with a string when testing things but when loading my full dataset I use map and it gets called with a string tensor
    if type(file_path) is not str:
        file_path = compat.as_text(file_path.numpy())   
    
    sentinel_data =  rasterio.open(file_path)

    red = sentinel_data.read(4)
    red = replace_outliers(red, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    green = sentinel_data.read(3)
    green = replace_outliers(green, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    blue = sentinel_data.read(2)
    blue = replace_outliers(blue, outlier_max_threshold, outlier_replacement_method, outlier_min_threshold)
    
    rgb = np.dstack((red, green, blue))   
    
    sentinel_data.close()
    
    return tf.convert_to_tensor(rgb)
                                             
def convert_sen1_product_to_tensor(file_path):
    #sentinel_data =  rasterio.open(file_path.numpy())
    file_path = compat.as_text(file_path.numpy()) 

    sentinel_data =  rasterio.open(file_path)
    vh = sentinel_data.read(1)
    vv = sentinel_data.read(2)
    
    sentinel_data.close()
    
    sen1 = np.dstack((vh,vv))   # could possibly replace this with tf.experimental.numpy.dstack
    #sen1 = tf.keras.utils.normalize(sen1)
    
    return tf.convert_to_tensor(sen1)

def process_path(file_path):
    
    sen2 = tf.py_function(func=convert_sen2_product_to_rgb_tensor, inp=[file_path], Tout=[tf.float32]) 
    sen2 = tf.reshape(sen2, [256, 256, 3])

    sen1_file_path = tf.strings.regex_replace(file_path, "/S2/", "/S1") # should there be another / after S1!!!
    sen1 = tf.py_function(func=convert_sen1_product_to_tensor, inp=[file_path], Tout=[tf.float32])
    sen1 = tf.reshape(sen1, [256, 256, 2])
    
    return sen1, sen2