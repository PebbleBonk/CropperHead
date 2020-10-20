from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import math
import glob


def show_single(img, cr, f=lambda x:x):
    """ Plot an image with the crop information as lines """
    if isinstance(img, str):
        im = Image.open(img)
    else:
        im = Image.fromarray(img)

    c = cr[0:4]
    r = cr[4]

    # HACK: apply function to plto correctly:
    c = [f(p) for p in c]

    #im = np.clip(ndimage.rotate(im, ci[4]), 0, 1)
    im = im.rotate(r)
    imw, imh = im.width, im.height

    plt.figure()
    plt.imshow(im)
    plt.title(f'L: {c[0]}, T: {c[1]}, R: {c[2]}, B: {c[3]}, A: {r}')

    plt.hlines([c[0]*imh, c[2]*imh], 0, imw, colors=['r', 'g'])
    plt.vlines([c[1]*imw, c[3]*imw], 0, imh, colors=['r', 'g'])

    plt.axis('off')
    plt.show()


def show_batch(image_batch, label_batch, f=lambda x:x):
    """ Show a batch of images (np.arrays) """
    g = math.ceil(math.sqrt(len(image_batch)))
    plt.figure(figsize=(15,15))
    for n in range(len(image_batch)):
        plt.subplot(g,g,n+1)
        img = image_batch[n]
        cr = label_batch[n]
        c = cr[0:4]
        r = cr[4]

        # HACK: apply function to plto correctly:
        c = [f(p) for p in c]

        if isinstance(img, str):
            im = Image.open(img)
        else:
            im = Image.fromarray((img*255).astype(np.uint8))

        imw, imh = im.width, im.height
        im = im.rotate(r)

        plt.imshow(im)
        plt.hlines([c[0]*imh, c[2]*imh], 0, imw, colors=['r', 'g'])
        plt.vlines([c[1]*imw, c[3]*imw], 0, imh, colors=['r', 'g'])
        plt.axis('off')


def prep_photo(photo_name, h, w):
    """ Prepare a photo to be propagated through the model """
    # Load image:
    im = Image.open(photo_name)
    # Resize image:
    im = im.resize((w, h))
    # Turn into numpy array:
    im = np.asarray(im).astype(np.float32)/255.
    # Reshape so it can be fed:
    im = np.expand_dims(im, axis=0)
    return im


def prep_dir(photo_dir, h, w):
    """ Prepare a directory of photos to be propagated through the model """
    photos = glob.glob(photo_dir+'*.jpg')
    res = None
    for p in photos:
        im = prep_photo(p, h, w)
        if res is None:
            res = im
        else:
            res = np.concatenate((res, im), axis=0)
    return photos, res


def plot_pred(img, cr, f=lambda x:x):
    c = cr[0:4]
    r = cr[4]

    # HACK: apply function to plto correctly:
    c = [f(p) for p in c]

    if isinstance(img, str):
        im = Image.open(img)
    else:
        im = Image.fromarray(img)
    im = im.rotate(r)
    imw = im.width
    imh = im.height

    plt.figure()
    plt.title(f'Rotation: {pred[4]}')
    imshow(np.asarray(im))
    plt.hlines([c[0]*imh, c[2]*imh], 0, imw, colors=['r', 'g'])
    plt.vlines([c[1]*imw, c[3]*imw], 0, imh, colors=['r', 'g'])
    plt.axis('off')
    plt.show()


