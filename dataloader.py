import torch
import torchio as tio
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
from torchio.transforms import (
    Crop,
    Resize,
    CropOrPad,
    Compose,
    CopyAffine,
    RescaleIntensity
)

class MedData_Load(torch.utils.data.Dataset):
    def __init__(self, flag, images_dir, labels_dir, fold_csv_dir, fold_arch, resize_size, fold):
        self.subjects = []

        fold_csv = pd.read_csv(fold_csv_dir)

        all_ids = list(fold_csv['Image_ID'].unique())
        # test_ids = list(fold_csv[fold_csv['Fold_ID'] == fold]['Image_ID'])
        # train_ids = list(set(all_ids) - set(test_ids))
        test_ids = all_ids
        train_ids = all_ids

        if flag == 'Train':
            required_id = train_ids
        if flag == 'Test':
            required_id = test_ids

        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob(fold_arch))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob(fold_arch))
        if flag == 'Train':
            self.path_lists = list(zip(self.image_paths, self.label_paths))
        if flag == 'Test':
            self.path_lists = list(self.image_paths)
        random.shuffle(self.path_lists)

        count = 0
        # for image in tqdm(self.path_lists): # Test
        for image,label in tqdm(self.path_lists): # Train
            ids = str(image).split('/')[-1].split('.')[0]
            #idl = str(label).split('/')[-1].split('-')[1]
            label = ('/data/labelnii160crop/Segmentation-{}-label.nii'.format(ids))
            
            img = tio.ScalarImage(image)

            lab = tio.LabelMap(label)           
            subject = tio.Subject(
                image = img,
                label = lab,
                image_shape = img.shape[1:],
                ID = ids
            )
            transform = Compose([
                CopyAffine('image'),
                Resize(resize_size),
                # RescaleIntensity((-1,1))
            ])
            subject=transform(subject)          
            img_tensor = subject['image'].data

            # Convert to numpy array for processing
            img_np = img_tensor.numpy()

            # Apply thresholding
            img_np[img_np > 3000] = 3000
            img_np[img_np < 300] = 300
            img_np = (img_np - 300) / (3000 - 300)  # Normalize between 0 and 1 (though it's already in this range)

            # Convert back to tensor
            subject['image'].data = torch.from_numpy(img_np).float()
            self.subjects.append(subject)
            count = count+1
        self.dataset = tio.SubjectsDataset(self.subjects)