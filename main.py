from segmentation_models_pytorch.utils.metrics import Accuracy
from segmentation_models_pytorch.utils.train import ValidEpoch, TrainEpoch

import dataset_utils

import albumentations as albu
import cv2
import numpy as np
import os
import random
import segmentation_models_pytorch as smp
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary
from torchmetrics import JaccardIndex
from torchviz import make_dot


CLASSES = [1, 2, 3] # ['Background', 'Nail', 'Primary Capillary', 'Secondary Capillary']

TRAIN_VAL_TEST_RATIO = (8, 1, 1)

DIR_PATH = os.path.dirname(__file__)
CSD_PATH = os.path.join(DIR_PATH, 'capillaries_seg_data')
DATA_DIR = os.path.join(CSD_PATH, 'ds0')
FRAMES_DIR = os.path.join(DATA_DIR, 'img')
MASKS_DIR = os.path.join(DATA_DIR, 'masks_machine')
NORM_MASKS_DIR = os.path.join(DATA_DIR, 'norm_masks')
MODELS_DIR = os.path.join(CSD_PATH, 'models')
BEST_MODEL = 'best_model.pth'
TRAIN_VAL_LOGS = 'train_val_logs.txt'

TRAIN_BATCH_SIZE = 4 # 32
VAL_BATCH_SIZE = 1 # 32
LR1 = 0.0001
LR2 = 1e-5
LR3 = 1e-6
NUM_OF_EPOCHS = 50 # 100 # 300

ENCODER = 'resnet34' # 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
LOSS = smp.losses.DiceLoss # smp.utils.losses.BCELoss()


def get_all_imgs(imgs_path):
    all_imgs = []
    if os.path.exists(imgs_path) and os.path.isdir(imgs_path) and len(os.listdir(imgs_path)) > 0 :
        for img in os.listdir(imgs_path):
            all_imgs.append(img)
    return all_imgs


def get_train_val_test_frames(frames_path, ratio=TRAIN_VAL_TEST_RATIO):
    all_frames = get_all_imgs(frames_path)
    all_frames_count = len(all_frames)
    ratio_sum = sum(ratio)
    train_count = all_frames_count * ratio[0] // ratio_sum
    val_test_count = all_frames_count - train_count
    if ratio[1] == ratio[2] and (val_test_count % 2) == 0:
        val_count = val_test_count // 2
        test_count = val_test_count // 2
    else:
        val_count = all_frames_count * ratio[1] // ratio_sum
        test_count = val_test_count - val_count

    train_frames = set(random.sample(all_frames, train_count))
    val_test_frames = set(all_frames) - train_frames
    val_frames = set(random.sample(list(val_test_frames), val_count))
    test_frames = val_test_frames - val_frames

    return train_frames, val_frames, test_frames


def save_augs(path, aug_dataset, augs_dir_name='augs', max_count=100):
    saved_frames = 0
    #if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
    #    dataset_utils.clean_dir(path)
    #else:

    aug_frames_dir = os.path.join(path, augs_dir_name, 'frames')
    aug_masks_dir = os.path.join(path, augs_dir_name, 'masks')
    os.makedirs(aug_frames_dir, exist_ok=True)
    os.makedirs(aug_masks_dir, exist_ok=True)
    dataset_utils.clean_dir(aug_frames_dir)
    dataset_utils.clean_dir(aug_masks_dir)

    for i in range(max_count):
        aug_frame, aug_mask = aug_dataset[i]
        aug_frame_path = os.path.join(aug_frames_dir, f'{i}.png')
        aug_mask_path = os.path.join(aug_masks_dir, f'{i}.png')
        frame_is_written = cv2.imwrite(aug_frame_path, aug_frame)
        mask_is_written = cv2.imwrite(aug_mask_path, aug_mask)
        """
        if frame_is_written:
            print(f'Aug frame {aug_frame_path} is successfully saved.')
        if mask_is_written:
            print(f'Aug mask {aug_mask_path} is successfully saved.')
        """
        saved_frames += 1

    return saved_frames


def make_model_dir(models_path, model_name, params=""):
    model_dir = os.path.join(models_path, model_name, params)
    os.makedirs(model_dir, exist_ok=True)
    dataset_utils.clean_dir(model_dir)

    return model_dir


def get_training_augmentation():
    train_transform = [
        #albu.HorizontalFlip(p=0.5),
        #albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.0, rotate_limit=180, shift_limit=0.0, p=1, border_mode=0)
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        ##albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class CapillariesSegDataset(Dataset):
    """

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            frames_dir,
            masks_dir,
            frames_names,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.frames_names = frames_names

        self.classes = classes

        self.frames_paths = [os.path.join(self.frames_dir, frame) for frame in self.frames_names]
        self.masks_paths = [os.path.join(self.masks_dir, mask) for mask in self.frames_names]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        frame = cv2.imread(self.frames_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks_paths[i], cv2.IMREAD_GRAYSCALE)

        # extract certain classes from mask (e.g. cars)
        ##masks = [(mask == v) for v in self.class_values]
        ##mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=frame, mask=mask)
            frame, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=frame, mask=mask)
            frame, mask = sample['image'], sample['mask']

        #return frame, mask
        return cv2.normalize(frame, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F), cv2.normalize(mask, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    def __len__(self):
        return len(self.frames_dir)


if __name__ == '__main__':
    dataset_utils.test_cuda()
    n_cpu = os.cpu_count()
    print(f'n_cpu: {n_cpu}')

    train_frames, val_frames, test_frames = get_train_val_test_frames(FRAMES_DIR)
    print(f'len(train_frames) = {len(train_frames)}, len(val_frames) = {len(val_frames)}, len(test_frames) = {len(test_frames)}')
    print(f'len(all_frames) = {len(train_frames.union(val_frames).union(test_frames))}')

    unet = smp.Unet( # encoder_depth=4
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=len(CLASSES),
        activation=ACTIVATION
    )

    pspnet = smp.PSPNet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=len(CLASSES),
        activation=ACTIVATION
    )
    
    fpn = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=len(CLASSES), 
        activation=ACTIVATION
    )
    
    linknet = smp.Linknet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=len(CLASSES), 
        activation=ACTIVATION
    )
    
    pan = smp.PAN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=1,
        classes=len(CLASSES), 
        activation=ACTIVATION
    )

    current_model = pan
    model_name = type(current_model).__name__

    train_aug_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        train_frames,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    val_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        val_frames,
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    train_loader = DataLoader(train_aug_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=n_cpu) # num_workers=8
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=n_cpu) # num_workers=4

    metrics = [
        Accuracy(threshold=0.5, activation=None)  # activation='sigmoid' # activation='tanh')
    ]

    optimizer = torch.optim.Adam([
        dict(params=current_model.parameters(), lr=LR1)
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = TrainEpoch( # model, loss, metrics, optimizer
        model=current_model,
        loss=LOSS,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True
    )



    model_path = make_model_dir(MODELS_DIR, model_name, f'{NUM_OF_EPOCHS}_{ACTIVATION}_{type(LOSS).__name__}_{type(optimizer).__name__}_{type(metrics[0]).__name__}')
    print(f'model_path: {model_path}')
    best_model_path = os.path.join(model_path, BEST_MODEL)

    saved_augs = save_augs(DATA_DIR, train_aug_dataset)
    print(f'saved_augs: {saved_augs}')

    """
    max_score = 0
    train_logs = {'accuracy': [], 'loss': []}
    valid_logs = {'accuracy': [], 'loss': []}

    for i in range(0, NUM_OF_EPOCHS):
        print(f'\nEpoch: {i + 1}')
        train_log = train_epoch.run(train_loader)
        valid_log = valid_epoch.run(val_loader)

        train_logs['accuracy'].append(train_log['accuracy'])
        valid_logs['accuracy'].append(valid_log['accuracy'])
        train_logs['loss'].append(train_log['bce_loss'])
        valid_logs['loss'].append(valid_log['bce_loss'])

        # do something (save model, change lr, etc.)
        if max_score < valid_log['accuracy']:
            max_score = valid_log['accuracy']
            torch.save(current_model, best_model_path)
            print('Model saved!')

        if i == int(NUM_OF_EPOCHS * 0.75):
            optimizer.param_groups[0]['lr'] = LR2
            print('Decrease decoder learning rate to 1e-5!')


    # load best saved checkpoint
    best_model = torch.load('./best_model.pth')

    # create test dataset
    test_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        test_frames,
        augmentation=None,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    test_dataloader = DataLoader(test_dataset)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=LOSS,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print(f'Test logs: {logs}\n')
    """








    """
    aug_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        train_frames,
        augmentation=get_training_augmentation(),
        classes=CLASSES
    )

    saved_augs = save_augs(DATA_DIR, aug_dataset)
    print(f'saved_augs: {saved_augs}')
    """

    """
    print(f'len(os.listdir(MASKS_DIR)) : {len(os.listdir(MASKS_DIR))}')
    os.makedirs(NORM_MASKS_DIR, exist_ok=True)
    dataset_utils.clean_dir(NORM_MASKS_DIR)
    for mask_file in os.listdir(MASKS_DIR):
        mask_file_path = os.path.join(MASKS_DIR, mask_file)
        print(f'mask_file_path: {mask_file_path}')
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR)
        print(f'np.unique(mask) : {np.unique(mask)}')
        mask[mask != 1] = 0
        norm_mask = cv2.normalize(mask, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        norm_mask_path = os.path.join(NORM_MASKS_DIR, mask_file)
        isWritten = cv2.imwrite(norm_mask_path, norm_mask)
        if isWritten:
            print(f'Norm {mask_file} is successfully saved as file.')
    """