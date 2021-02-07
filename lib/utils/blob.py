from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def im_list_to_blob(ims):

    def get_reflectance(image): 
        l = 0.114*image[:,:,0]+0.587*image[:,:,1]+0.299*image[:,:,2]+0.00001
        #plt.imshow(l1)
        '''
        plt.imshow(image)
        plt.imshow(l_cap)
        alpha = 0.015
        l_cap1= l_cap.flatten()
        n = len(l_cap1)
        l_cap_vec = l_cap.reshape((n,1))
        I = np.identity(n)
        filter_x = np.array([[0,0,0],[0,-1,1],[0,0,0]])
        filter_y = np.array([[0,0,0],[0,-1,0],[0,1,0]])
        l_cap_x = filter_image(l_cap,filter_x)
        l_cap_y = filter_image(l_cap,filter_y)
        eps = 1
        A_x = alpha/(np.abs(l_cap_x)+eps)
        A_y = alpha/(np.abs(l_cap_y)+eps)
        a_x = A_x.flatten()
        a_y = A_y.flatten()
        a_x = np.diag(a_x)
        a_y = np.diag(a_y)
        l = np.matmul(np.linalg.inv(I+a_x+a_y),l_cap_vec)
        '''
        r = np.zeros(image.shape)
        r[:,:,0] =np.divide(image[:,:,0],l)
        r[:,:,1] =np.divide(image[:,:,1],l)
        r[:,:,2] =np.divide(image[:,:,2],l)
        m = np.max(r)
        n = np.min(r)
        d = m-n
        #r = np.divide(r,d)
        #plt.imshow(r)
        return (r*255).astype(int)
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def im_list_to_reflect_blob(ims):
    """Convert a list of images into a reflectance map blob.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    def get_reflectance(image): 
        l = 0.114*image[:,:,0]+0.587*image[:,:,1]+0.299*image[:,:,2]+0.00001
        #plt.imshow(l1)
        '''
        plt.imshow(image)
        plt.imshow(l_cap)
        alpha = 0.015
        l_cap1= l_cap.flatten()
        n = len(l_cap1)
        l_cap_vec = l_cap.reshape((n,1))
        I = np.identity(n)
        filter_x = np.array([[0,0,0],[0,-1,1],[0,0,0]])
        filter_y = np.array([[0,0,0],[0,-1,0],[0,1,0]])
        l_cap_x = filter_image(l_cap,filter_x)
        l_cap_y = filter_image(l_cap,filter_y)
        eps = 1
        A_x = alpha/(np.abs(l_cap_x)+eps)
        A_y = alpha/(np.abs(l_cap_y)+eps)
        a_x = A_x.flatten()
        a_y = A_y.flatten()
        a_x = np.diag(a_x)
        a_y = np.diag(a_y)
        l = np.matmul(np.linalg.inv(I+a_x+a_y),l_cap_vec)
        '''
        r = np.zeros(image.shape)
        r[:,:,0] =np.divide(image[:,:,0],l)
        r[:,:,1] =np.divide(image[:,:,1],l)
        r[:,:,2] =np.divide(image[:,:,2],l)
        #plt.imshow(r)
        return r
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = get_reflectance(ims[i])
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
