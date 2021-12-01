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
    
    cur_best_psnr_score = 0
    cur_best_fid_score = 0
    cur_best_ssim_score = 0
    cur_best_blur_score = 0
    
    
    cur_worst_psnr_score = 0
    cur_worst_fid_score = 0
    cur_worst_ssim_score = 0
    cur_worst_blur_score = 0
    
    psnr_scores = np.array([])
    mse_scores = np.array([])
    fid_scores = np.array([])
    ssim_scores = np.array([])
    blur_scores = np.array([])
    

    for cur_images in test_ds:
        sen1 = cur_images[0]
        # get real image with values between 0 and 1 (have to add 1 then divide by 2 as previously scaled data from -1 to 1 for neural network)
        real_sen2 = (cur_images[1]+1)/2
        
        # generate fake image using generator and convert the values to between 0 and 1 
        # need to call expand dims as generator is used to receiving batches of images rather than a single image and similarly call squeeze as I get a batch of 1 back but just need image
        fake_sen2 = tf.squeeze(model.generator.predict(tf.expand_dims(sen1, axis=0)))
        fake_sen2 = (fake_sen2+1)/2
        
        grey_fake = tf.image.rgb_to_grayscale(fake_sen2) # greyscale version of fake image using for calculating image sharpness and brightness
        
        psnr = tf.image.psnr(real_sen2, fake_sen2, max_val=1.0).numpy()
       
        
        if psnr > cur_best_psnr_score:
            cur_best_psnr_score = psnr
            cur_best_psnr_sen1 = sen1
            cur_best_psnr_sen2 = real_sen2
            cur_best_psnr_fake = fake_sen2 
        
        if psnr < cur_worst_psnr_score or cur_worst_psnr_score == 0:
            cur_worst_psnr_score = psnr
            cur_worst_psnr_sen1 = sen1
            cur_worst_psnr_sen2 = real_sen2
            cur_worst_psnr_fake = fake_sen2 
        psnr_scores = np.append(psnr_scores, psnr)
        
        
        blur = blur_score(grey_fake)
        
        if blur > cur_best_blur_score:
            cur_best_blur_score = blur
            cur_best_blur_sen1 = sen1
            cur_best_blur_sen2 = real_sen2
            cur_best_blur_fake = fake_sen2 
        
        if blur < cur_worst_psnr_score or cur_worst_psnr_score == 0:
            cur_worst_blur_score = blur
            cur_worst_blur_sen1 = sen1
            cur_worst_blur_sen2 = real_sen2
            cur_worst_blur_fake = fake_sen2 
        
        blur_scores = np.append(blur_scores, blur)
        """
        mse = tf.keras.metrics.mean_squared_error(real_sen2, fake_sen2).numpy()
        tf.print(mse)
        if mse > cur_best_mse_score:
            cur_best_mse_score = psnr
            cur_best_mse_sen1 = sen1
            cur_best_mse_sen2 = real_sen2
            cur_best_mse_fake = fake_sen2 
        
        if mse < cur_worst_mse_score or cur_worst_mse_score == 0:
            cur_worst_mse_score = psnr
            cur_worst_mse_sen1 = sen1
            cur_worst_mse_sen2 = real_sen2
            cur_worst_mse_fake = fake_sen2 
        
        mse_scores = np.append(mse_scores, mse)
        """
        ssim = tf.image.ssim(real_sen2, fake_sen2, max_val=1.0).numpy()
        # get SSIM
        if ssim > cur_best_ssim_score:
            cur_best_ssim_score = ssim
            cur_best_ssim_sen1 = sen1
            cur_best_ssim_sen2 = real_sen2
            cur_best_ssim_fake = fake_sen2 
        
        if ssim < cur_worst_ssim_score or cur_worst_ssim_score == 0:
            cur_worst_ssim_score = ssim
            cur_worst_ssim_sen1 = sen1
            cur_worst_ssim_sen2 = real_sen2
            cur_worst_ssim_fake = fake_sen2 
        
        ssim_scores = np.append(ssim_scores, ssim)
        """
        # get FID
        if fid > cur_best_fid_score:
            cur_best_fid_score = psnr
            cur_best_fid_sen1 = sen1
            cur_best_fid_sen2 = real_sen2
            cur_best_fid_fake = fake_sen2 
        
        if psnr < cur_worst_fid_score:
            cur_worst_fid_score = psnr
            cur_worst_fid_sen1 = sen1
            cur_worst_fid_sen2 = real_sen2
            cur_worst_fid_fake = fake_sen2 
        
        np.append(fid_scores, fid)
        """
    """
    mse_avg = np.average(mse_scores)
    mse_std = np.std(mse_scores)  
    
    tf.print("best mse score", cur_best_mse_score)
    tf.print("worst mse score", cur_worst_mse_score)
    tf.print("mse average", mse_avg)
    tf.print("mse st deviation", mser_std)
          
    # plot and save best/worst images
    filename = experiment_name + "_best_mse.jpg"
    plot_images("Best PSNR score " + str(cur_best_mse_score), cur_best_mse_sen1, cur_best_mse_sen2, cur_best_mse_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_psnr.jpg"
    plot_images("Worst mse score " + str(cur_worst_mse_score), cur_worst_mse_sen1, cur_worst_mse_sen2, cur_worst_mse_fake, filename.replace(" ", "_"))
    """    
    psnr_avg = np.average(psnr_scores)
    psnr_std = np.std(psnr_scores)  
    
    tf.print("best score", cur_best_psnr_score)
    tf.print("worst score", cur_worst_psnr_score)
    tf.print("psnr average", psnr_avg)
    tf.print("psnr st deviation", psnr_std)
          
    # plot and save best/worst images
    filename = experiment_name + "_best_psnr"
    plot_images("Best PSNR score " + str(cur_best_psnr_score), cur_best_psnr_sen1, cur_best_psnr_sen2, cur_best_psnr_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_psnr"
    plot_images("Worst PSNR score " + str(cur_worst_psnr_score), cur_worst_psnr_sen1, cur_worst_psnr_sen2, cur_worst_psnr_fake, filename.replace(" ", "_"))
    
    ssim_avg = np.average(ssim_scores)
    ssim_std = np.std(ssim_scores)  
    
    tf.print("best score", cur_best_ssim_score)
    tf.print("worst score", cur_worst_ssim_score)
    tf.print("psnr average", ssim_avg)
    tf.print("psnr st deviation", ssim_std)
          
    # plot and save best/worst images
    filename = experiment_name + "_best_ssim"
    plot_images("Best SSIM score " + str(cur_best_ssim_score), cur_best_ssim_sen1, cur_best_ssim_sen2, cur_best_ssim_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_ssim"
    plot_images("Worst SSIM score " + str(cur_worst_ssim_score), cur_worst_ssim_sen1, cur_worst_ssim_sen2, cur_worst_ssim_fake, filename.replace(" ", "_"))
    
    blur_avg = np.average(blur_scores)
    blur_std = np.std(blur_scores)  
    
    tf.print("Best blur scrore", cur_best_blur_score)
    tf.print("worst blur score", cur_worst_blur_score)
    tf.print("average blur", blur_avg)
    tf.print("blur st deviation", blur_std)
    
     # plot and save best/worst images
    filename = experiment_name + "_best_blur"
    plot_images("Least blurry image: " + str(cur_best_blur_score), cur_best_blur_sen1, cur_best_blur_sen2, cur_best_blur_fake, filename.replace(" ", "_"))
    
    filename = experiment_name + "_worst_ssim"
    plot_images("Blurriest image: " + str(cur_worst_blur_score), cur_worst_blur_sen1, cur_worst_blur_sen2, cur_worst_blur_fake, filename.replace(" ", "_"))
    # return metrics eg best worst fid,sim,psrn, average, std deviation?
        
        
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
    plt.show()
    plt.savefig("saved_images/" + filename)
    