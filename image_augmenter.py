import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage 
from PIL import Image
import numpy as np

def bb_to_crop(bb, a=None):
#     crop = [bb.x1, bb.y1, bb.x2, bb.y2]
    crop = bb.to_xyxy_array()
    if a is not None:
        crop = np.append(crop, a)
    return crop


# Inpired by:
# https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
class CropDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, x1_col, y1_col, x2_col, y2_col, a_col, dim, 
                 batch_size=32,  as_floats=False, zero_mean=True, shuffle=True, augment=False,
                 flipV=0.5, flipH=0.5, rotate=0.5, n_others=0, p_other=0.5):

        # Make sure the df we get has default index
        df.reset_index(drop=True, inplace=True)
        
        self.augment = augment
        self.as_floats = as_floats
        self.zero_mean = zero_mean
        self.flipH = flipH
        self.flipV = flipV
        self.rotate = rotate
        self.n_others = n_others
        self.p_other = p_other
        self.batch_size = batch_size
        self.image_paths = df[x_col]
        self.angles = df[a_col]
        self.indices = df.index.tolist()
        
        print(df.shape, min(self.indices), max(self.indices), "LEN", len(self))
        
        self.shuffle = shuffle
        self.x_col = x_col
        self.dim = dim
        self.bbs = []
        self.resize = iaa.Resize(self.dim)
        
        for i in self.indices:
            bb = np.asarray([df[x1_col][i], df[y1_col][i], df[x2_col][i], df[y2_col][i]])
            # Optionally expect and export relative values as bounding boxes:
            if as_floats:
                 # bb = [c*self.dim for c in bb] # Convert from pixels to percentages
                if self.zero_mean:
                    bb = (bb+1)/2
                bb = bb*self.dim
            self.bbs.append(BoundingBoxesOnImage.from_xyxy_array(bb.reshape((1,4)), (self.dim, self.dim))) 
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # select data and load images
        images = [np.asarray(Image.open(self.image_paths[k])) for k in batch]
        images = [ np.float32(im/255) for im in images]
        crops = [self.bbs[k] for k in batch]
        angles = [self.angles[k] for k in batch]

        # preprocess and augment data
        if self.augment == True:
            images, crops = self.augmentor(images, crops)
        # Convert bounding boxes back to vectors, add angle info:
        labels = np.asarray([bb_to_crop(bb, self.angles[i]) for i, bb in enumerate(crops)])
        
        
        # if using percentage, convert back to vectors:
        if self.as_floats:
            labels[:,:4] = labels[:,:4]/self.dim
            # To zero mean if necessary:
            if self.zero_mean:
                labels[:,:4] = labels[:,:4]*2-1
        
        # Apply optional preprocessing to the images:
        # images = np.array([preprocess_input(img) for img in images])
        images = np.asarray(self.resize(images=images))
        # Convert back to 
        return images, labels

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
            
    def augmentor(self, images, bbs):
        """ Apply data augmentation

            Randomly:
                - Rotate images 90 CW/CCW
                - Flip Horizontal/Vertical

            Also remember to rotate/flip the labels accrodingly:
                - Rotate 90 deg CW: shift values
                - Rotate 90 CCW: shift values, negate them
                - Flip Horizontal: negate left, right
                - Flip vertical: negate top, bottom
        """
        # The probability of conducting the other, less important augmentations:
        sometimes = lambda aug: iaa.Sometimes(self.p_other, aug)

        seq = iaa.Sequential(
            [
            # apply the following augmenters to most images
            iaa.Fliplr(self.flipH),  # horizontally flip 50% of all images
            iaa.Flipud(self.flipV),  # vertically flip 20% of all images
            iaa.Sometimes(self.rotate, iaa.geometric.Rot90(k=[1,2,3])),  # rotate by either -90 or 90 deg

            # execute 0 to n_others of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, self.n_others),
                [
                    # convert images into their superpixel representation
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

                    iaa.OneOf([
                            iaa.GaussianBlur((0, 1.0)),
                            # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(3, 5)),
                            # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 5)),
                            # blur image using local medians with kernel sizes between 2 and 7
                    ]),

                    # sharpen images
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                    # emboss images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # add gaussian noise to images
                    iaa.AdditiveGaussianNoise(loc=0,
                                                scale=(0.0, 0.01 * 255),
                                                per_channel=0.5),

                    iaa.OneOf([
                            iaa.Dropout((0.01, 0.05), per_channel=0.5),
                            # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.01, 0.03),
                                                size_percent=(0.01, 0.02),
                                                per_channel=0.2),
                    ]),

                    # invert color channels
                    iaa.Invert(0.01, per_channel=True),

                    # change brightness of images (by -10 to 10 of original value)
                    iaa.Add((-2, 2), per_channel=0.5),

                    # change hue and saturation
                    iaa.AddToHueAndSaturation((-1, 1)),

                    # move pixels locally around (with random strengths)
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                        sigma=0.25)),

                    # sometimes move parts of the image around
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
            ],
            random_order=True
        )
        return seq(images=images, bounding_boxes=bbs)

