import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import tensorflow.keras.metrics

# takes a greyscale image and calculates blurriness by applying a Laplacian filter then calculating variance
def blur_score(grey_img):
    # apply filter
    filter_size = tf.constant(3)
    laplace = tfio.experimental.filter.laplacian(tf.expand_dims(grey_img, axis=0), filter_size)
    return tf.math.reduce_std(laplace).numpy()

def eval_test_set(model, test_ds, experiment_name):
    
    metrics = {
        "cur_best_psnr_score": 0,
        "cur_best_ssim_score": 0,
        "cur_best_ms_ssim_score": 0,
        "cur_best_blur_score": 0,
        "cur_brightest_score": 0,
        "cur_worst_psnr_score": 0,
        "cur_worst_ssim_score": 0,
        "cur_worst_ms_ssim_score": 0,
        "cur_worst_blur_score": 0,
        "cur_darkest_score": 0 }
    
    psnr_scores = np.array([])
    ssim_scores = np.array([])
    ms_ssim_scores = np.array([])
    blur_scores = np.array([])
    brightness_scores = np.array([])
    

    for cur_images in test_ds:
        sen1 = cur_images[0]
        # get real image with values between 0 and 1 (have to add 1 then divide by 2 as previously scaled data from -1 to 1 for neural network)
        real_sen2 = (cur_images[1]+1)/2
        
        # generate fake image using generator and convert the values to between 0 and 1 
        # need to call expand dims as generator is used to receiving batches of images rather than a single image and similarly call squeeze as I get a batch of 1 back but just need image
        fake_sen2 = tf.squeeze(model.generator.predict(tf.expand_dims(sen1, axis=0)))
        fake_sen2 = (fake_sen2+1)/2
        
        grey_fake = tf.image.rgb_to_grayscale(fake_sen2) # greyscale version of fake image using for calculating image sharpness and brightness
        
        brightness = tf.math.reduce_mean(grey_fake).numpy()
        if brightness > metrics["cur_brightest_score"]:
            metrics["cur_brightest_score"] = brightness
            cur_brightest_sen1 = sen1
            cur_brightest_sen2 = real_sen2
            cur_brightest_fake = fake_sen2 
        
        if brightness < metrics["cur_darkest_score"] or metrics["cur_darkest_score"] == 0:
            metrics["cur_darkest_score"] = brightness
            cur_darkest_sen1 = sen1
            cur_darkest_sen2 = real_sen2
            cur_darkest_fake = fake_sen2 
        brightness_scores = np.append(brightness_scores, brightness)
        
        
        psnr = tf.image.psnr(real_sen2, fake_sen2, max_val=1.0).numpy()
       
        
        if psnr > metrics["cur_best_psnr_score"]:
            metrics["cur_best_psnr_score"] = psnr
            cur_best_psnr_sen1 = sen1
            cur_best_psnr_sen2 = real_sen2
            cur_best_psnr_fake = fake_sen2 
        
        if psnr < metrics["cur_worst_psnr_score"] or metrics["cur_worst_psnr_score"] == 0:
            metrics["cur_worst_psnr_score"] = psnr
            cur_worst_psnr_sen1 = sen1
            cur_worst_psnr_sen2 = real_sen2
            cur_worst_psnr_fake = fake_sen2 
        psnr_scores = np.append(psnr_scores, psnr)
        
        
        blur = blur_score(grey_fake)
        
        if blur > metrics["cur_best_blur_score"]:
            metrics["cur_best_blur_score"] = blur
            cur_best_blur_sen1 = sen1
            cur_best_blur_sen2 = real_sen2
            cur_best_blur_fake = fake_sen2 
        
        if blur < metrics["cur_worst_psnr_score"] or metrics["cur_worst_psnr_score"] == 0:
            metrics["cur_worst_blur_score"] = blur
            cur_worst_blur_sen1 = sen1
            cur_worst_blur_sen2 = real_sen2
            cur_worst_blur_fake = fake_sen2 
        
        blur_scores = np.append(blur_scores, blur)
       
        ssim = tf.image.ssim(real_sen2, fake_sen2, max_val=1.0).numpy()
        # get SSIM
        if ssim > metrics["cur_best_ssim_score"]:
            metrics["cur_best_ssim_score"] = ssim
            cur_best_ssim_sen1 = sen1
            cur_best_ssim_sen2 = real_sen2
            cur_best_ssim_fake = fake_sen2 
        
        if ssim < metrics["cur_worst_ssim_score"] or metrics["cur_worst_ssim_score"] == 0:
            metrics["cur_worst_ssim_score"] = ssim
            cur_worst_ssim_sen1 = sen1
            cur_worst_ssim_sen2 = real_sen2
            cur_worst_ssim_fake = fake_sen2 
        
        ssim_scores = np.append(ssim_scores, ssim)
        
        ms_ssim = tf.image.ssim_multiscale(real_sen2, fake_sen2, max_val=1.0).numpy()
        # get SSIM
        if ssim > metrics["cur_best_ms_ssim_score"]:
            metrics["cur_best_ms_ssim_score"] = ms_ssim
            cur_best_ms_ssim_sen1 = sen1
            cur_best_ms_ssim_sen2 = real_sen2
            cur_best_ms_ssim_fake = fake_sen2 
        
        if ssim < metrics["cur_worst_ms_ssim_score"] or metrics["cur_worst_ms_ssim_score"] == 0:
            metrics["cur_worst_ms_ssim_score"] = ms_ssim
            cur_worst_ms_ssim_sen1 = sen1
            cur_worst_ms_ssim_sen2 = real_sen2
            cur_worst_ms_ssim_fake = fake_sen2 
        
        ms_ssim_scores = np.append(ms_ssim_scores, ms_ssim)
    
    metrics["brightness_avg"] = np.average(brightness_scores)
    metrics["brightness_std"] = np.std(brightness_scores)  
    
    tf.print("highest brightness score", metrics["cur_brightest_score"])
    tf.print("worst score", metrics["cur_darkest_score"])
    tf.print("brightness average", metrics["brightness_avg"])
    tf.print("brightness deviation", metrics["brightness_std"])
          
    # plot and save best/worst images
    filename = experiment_name + "_brightest_image"
    plot_images("Brightest Image " + str(metrics["cur_brightest_score"]), cur_brightest_sen1, cur_brightest_sen2, cur_brightest_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_darkest_image"
    plot_images("Darkest Image " + str(metrics["cur_darkest_score"]), cur_darkest_sen1, cur_darkest_sen2, cur_darkest_fake, filename.replace(" ", "_"))
    
    
    metrics["psnr_avg"] = np.average(psnr_scores)
    metrics["psnr_std"] = np.std(psnr_scores)  
    
    tf.print("best score", metrics["cur_best_psnr_score"])
    tf.print("worst score", metrics["cur_worst_psnr_score"])
    tf.print("psnr average", metrics["psnr_avg"])
    tf.print("psnr st deviation", metrics["psnr_std"])
          
    # plot and save best/worst images
    filename = experiment_name + "_best_psnr"
    plot_images("Best PSNR score " + str(metrics["cur_best_psnr_score"]), cur_best_psnr_sen1, cur_best_psnr_sen2, cur_best_psnr_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_psnr"
    plot_images("Worst PSNR score " + str(metrics["cur_worst_psnr_score"]), cur_worst_psnr_sen1, cur_worst_psnr_sen2, cur_worst_psnr_fake, filename.replace(" ", "_"))
    
    metrics["ssim_avg"] = np.average(ssim_scores)
    metrics["ssim_std"] = np.std(ssim_scores)  
    
    tf.print("best score", metrics["cur_best_ssim_score"])
    tf.print("worst score", metrics["cur_worst_ssim_score"])
    tf.print("ssim average", metrics["ssim_avg"])
    tf.print("ssim st deviation", metrics["ssim_std"])
          
    # plot and save best/worst images
    filename = experiment_name + "_best_ssim"
    plot_images("Best SSIM score " + str(metrics["cur_best_ssim_score"]), cur_best_ssim_sen1, cur_best_ssim_sen2, cur_best_ssim_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_ssim"
    plot_images("Worst SSIM score " + str(metrics["cur_worst_ssim_score"]), cur_worst_ssim_sen1, cur_worst_ssim_sen2, cur_worst_ssim_fake, filename.replace(" ", "_"))
    
    metrics["ms_ssim_avg"] = np.average(ms_ssim_scores)
    metrics["ms_ssim_std"] = np.std(ms_ssim_scores)  
    
    tf.print("best ms ssim score", metrics["cur_best_ms_ssim_score"])
    tf.print("worst score", metrics["cur_worst_ms_ssim_score"])
    tf.print("psnr average", metrics["ms_ssim_avg"])
    tf.print("psnr st deviation", metrics["ms_ssim_std"])
          
    # plot and save best/worst images
    filename = experiment_name + "_best_ms_ssim"
    plot_images("Best Multiscale SSIM score " + str(metrics["cur_best_ms_ssim_score"]), cur_best_ms_ssim_sen1, cur_best_ms_ssim_sen2, cur_best_ms_ssim_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_ms_ssim"
    plot_images("Worst SSIM score " + str(metrics["cur_worst_ms_ssim_score"]), cur_worst_ms_ssim_sen1, cur_worst_ms_ssim_sen2, cur_worst_ms_ssim_fake, filename.replace(" ", "_"))
    
    metrics["blur_avg"] = np.average(blur_scores)
    metrics["blur_std"] = np.std(blur_scores)  
    
    tf.print("Best blur scrore", metrics["cur_best_blur_score"])
    tf.print("worst blur score", metrics["cur_worst_blur_score"])
    tf.print("average blur", metrics["blur_avg"])
    tf.print("blur st deviation", metrics["blur_std"])
    
     # plot and save best/worst images
    filename = experiment_name + "_best_blur"
    plot_images("Least blurry image: " + str(metrics["cur_best_blur_score"]), cur_best_blur_sen1, cur_best_blur_sen2, cur_best_blur_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_ssim"
    plot_images("Blurriest image: " + str(metrics["cur_worst_blur_score"]), cur_worst_blur_sen1, cur_worst_blur_sen2, cur_worst_blur_fake, filename.replace(" ", "_"))
    # return metrics eg best worst fid,sim,psrn, average, std deviation?
    
    return metrics
        
def plot_images(title, sen1, real_sen2, fake_sen2, filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)
    ax1.axis('off')
    ax1.imshow(sen1)
    ax1.set_title("Normalized Sentinel 1")
    ax2.axis('off')
    ax2.imshow(real_sen2)
    ax2.set_title("Real Sentinel 2 Image")
    ax3.axis('off')
    ax3.imshow(fake_sen2)
    ax3.set_title("Generated Sentinel 2 Image")
    plt.savefig("saved_images/" + filename)
    plt.show()
    
    