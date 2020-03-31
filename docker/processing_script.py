import numpy as np
import os, sys
from glob import glob
import imageio
from sklearn.model_selection import train_test_split
import nibabel as nib
import json

    
input_data_path = '/opt/ml/processing/input'
output_data_path = '/opt/ml/processing/output'

def create_dir_if_not_exist(d):
    if not os.path.exists(d):
        os.makedirs(d)

def process_mris(images, masks, target_dir, mask_dir):
    for idx, f in enumerate(images):
        # print(f)
        nii = nib.load(f)
        mris = nii.get_fdata().squeeze()[np.arange(1, nii.shape[0], 2), :, :]
        mris = mris / mris.max() * 255
        
        nii_mask = nib.load(masks[idx])
        segmentations = nii_mask.get_fdata().squeeze()[np.arange(1, nii.shape[0], 2), :, :]
        
        # dimension check
        if nii.shape != nii_mask.shape:
            print('Dimension of image and mask do not match. %s %s' % (nii.shape, nii_mask.shape))
            continue
            
        num_slices = mris.shape[0]
        for i in range(num_slices):
            new_fname = "_".join(os.path.basename(f).split('.')[0].split('_')[:8])+"_%i.png" % i
            
            mri_rotate = np.rot90(mris[i]).astype(np.uint8)
            segmentation_rotate = np.rot90(segmentations[i]).astype(np.uint8)
            
            # print('%s: %d' % (new_fname, mri_rotate.max()))
            
            imageio.imsave(os.path.join(target_dir, new_fname), mri_rotate)
            imageio.imsave(os.path.join(mask_dir, new_fname), segmentation_rotate)
            

def main():
    # get image lists
    # print('Listing the input_data_path')
    # print(os.listdir(input_data_path))
    images = glob(os.path.join(input_data_path, 'brain_mri','disc1','*','*','*','*','*t88_gfc.img'))
    indices = [i.split('/')[-5] for i in images]
    masks = [glob(os.path.join(input_data_path, 'brain_mri','disc1', i, 'FSL_SEG', '%s_*_t88_masked_gfc_fseg.img' % i))[0]
             for i in indices]
    
    # print('inside images:')
    # print(len(images))
    # print(images[0:5])
    train_images, validation_images, train_masks, validation_masks = train_test_split(
                                                        images, masks, test_size=0.2, random_state=1984)
    
    # creating directory structure
    train_dir = os.path.join(output_data_path, 'train')
    validation_dir = os.path.join(output_data_path, 'validation')

    train_mask_dir = os.path.join(output_data_path, 'train_annotation')
    validation_mask_dir = os.path.join(output_data_path, 'validation_annotation')

    create_dir_if_not_exist(train_dir)
    create_dir_if_not_exist(validation_dir)
    create_dir_if_not_exist(train_mask_dir)
    create_dir_if_not_exist(validation_mask_dir)

    # preprocess images
    process_mris(train_images, train_masks, train_dir, train_mask_dir)
    process_mris(validation_images, validation_masks, validation_dir, validation_mask_dir)
    
    label_map = { "scale": 1 }
    label_dir = os.path.join(output_data_path, 'label_map')
    create_dir_if_not_exist(label_dir)

    with open(os.path.join(label_dir, 'train_label_map.json'), 'w') as lm_fname:
        json.dump(label_map, lm_fname)

    
if __name__=='__main__':
    main()