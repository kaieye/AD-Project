from subprocess import call
from glob import glob

path = '/root/autodl-tmp/ADNI_processed_nii'
root_folder = '/root/autodl-tmp'

processed = glob('/root/autodl-tmp/seg/*.nii')
for file in glob(path + '/*.nii'):
    name = file.split('/')[-1]
    if file.replace('ADNI_processed_nii','seg') in processed:
        print('processed and continue')
        continue
    call('bash segmentation.sh ' + path + ' ' + name + ' ' + root_folder, shell=True)
# import csv
# import nibabel as nib
# import numpy as np
# from tqdm import tqdm
#
# with open('/home/sq/ssl-brain/lookupcsv/CrossValid/exp4/test.csv', 'r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in tqdm(reader):
#         data = np.load(row['path'] + row['filename'])
#         data = nib.Nifti1Image(data, affine=np.eye(4))
#         nib.save(data, '/data_2/sq/NACC_ALL_seg/mri4/' + row['filename'].replace('.npy', '.nii'))