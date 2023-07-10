
import dataset_utils

import albumentations as albu
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary
import torchmetrics
from torchmetrics import JaccardIndex
from torchviz import make_dot
import torchvision
from torchvision import transforms, models


CLASSES = (1, 2, 3) # ('Background', 'Nail', 'Primary Capillary', 'Secondary Capillary')

#CLASSES = (1, 2) # ('Background', 'Nail', 'Capillary')

TRAIN_VAL_TEST_RATIO = (8, 1, 1)

DIR_PATH = os.path.dirname(__file__)
CSD_PATH = os.path.join(DIR_PATH, 'capillaries_seg_data')
DATA_DIR = os.path.join(CSD_PATH, 'ds0')
FRAMES_DIR = os.path.join(DATA_DIR, 'img')
MASKS_DIR = os.path.join(DATA_DIR, 'masks_machine')
NORM_MASKS_DIR = os.path.join(DATA_DIR, 'norm_masks')
MODELS_DIR = os.path.join(CSD_PATH, 'models')
MHE_MASKS_DIR = os.path.join(MODELS_DIR, 'masks')
BEST_MODEL = 'best_model.pth'
TRAIN_VAL_LOGS = 'train_val_logs.txt'

TRAIN_BATCH_SIZE = 4 # 32
VAL_BATCH_SIZE = 1 # 32
LR1 = 0.0001
LR2 = 1e-5
LR3 = 1e-6
NUM_OF_EPOCHS = 1 # 50 # 100 # 300
FRAME_RESIZE = (1024, 1024)

#ENCODER = 'resnet34' # 'se_resnext50_32x4d'
#ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda:0'
LOSS = torch.nn.CrossEntropyLoss() # smp.utils.losses.BCELoss()


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


def save_augs(path, aug_dataset, augs_dir_name='augs', max_count=128):
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

    for i in range(len(aug_dataset)):
        if i < max_count:
            aug_frame, aug_mask = aug_dataset[i]
            print(f'aug_frame.shape : {aug_frame.shape}')
            print(f'aug_mask.shape : {aug_mask.shape}')
            #print(f'np.unique(aug_frame, return_counts=True): {np.unique(aug_frame, return_counts=True)}')
            #print(f'np.unique(aug_mask, return_counts=True): {np.unique(aug_mask, return_counts=True)}')
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


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


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
    return x.transpose(2, 0, 1).astype(np.float32)


#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


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
        #transforms.ToTensor(),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def save_summary(model, path=MODELS_DIR, filename='summary.txt'):
    if os.path.exists(path) and os.path.isdir(path):
        model_stats = summary(model, (1, 3, FRAME_RESIZE[0], FRAME_RESIZE[1]), verbose=1)
        summary_str = str(model_stats)
        summary_file_path = os.path.join(path, filename)
        with open(summary_file_path, 'w', encoding="utf-8") as f:
            f.write(summary_str)

        x = torch.randn(1, 3, FRAME_RESIZE[0], FRAME_RESIZE[1]).to(DEVICE)
        MyConvNetVis = make_dot(model(x), params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True)
        MyConvNetVis.format = "png"
        MyConvNetVis.directory = path
        try:
            #MyConvNetVis.view()
            MyConvNetVis.save()
        except Exception as e:
            print('torchviz.make_dot.save() fail: %s' % e)

        del model_stats, summary_str, MyConvNetVis
    else:
        print(f'{path} doesnt exist!')


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

        frame = cv2.resize(frame, FRAME_RESIZE, interpolation=cv2.INTER_AREA) # cv2.INTER_LINEAR)
        mask = cv2.resize(mask, FRAME_RESIZE, interpolation=cv2.INTER_NEAREST) # cv2.INTER_AREA)

        frame = cv2.normalize(frame, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ##mask = mask.astype(np.float32)
        # mask = cv2.normalize(mask, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        frame = self.expand_greyscale_image_channels(frame)
        mask = self.mask_multi_hot_encoding(mask)
        #self.save_mask(mask, self.frames_names[i], MHE_MASKS_DIR)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=frame, mask=mask)
            frame, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=frame, mask=mask)
            frame, mask = sample['image'], sample['mask']

        return frame, mask

    def __len__(self):
        return len(self.frames_names)

    def save_mask(self, mhe_mask, name, masks_dir):
        submasks_paths = []
        if not (os.path.exists(masks_dir) and os.path.isdir(masks_dir)):
            os.makedirs(masks_dir, exist_ok=True)

        for i in range(mhe_mask.shape[2]):
            submask = mhe_mask[:, :, i]
            submask_file = os.path.join(masks_dir, f'{name}_{i}.png')
            #print(f'np.unique(submask): {np.unique(submask)}')
            cv2.imwrite(submask_file, submask)
            submasks_paths.append(submask_file)

        return submasks_paths

    def expand_greyscale_image_channels(self, grey_image_arr):
        grey_image_arr = np.expand_dims(grey_image_arr, 2)
        grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=2)
        return grey_image_arr_3_channel


    def mask_multi_hot_encoding(self, mask_tensor):

        nail_mask = np.copy(mask_tensor)
        pc_mask = np.copy(mask_tensor)
        sc_mask = np.copy(mask_tensor)

        nail_mask[nail_mask != CLASSES[0]] = 0
        nail_mask[nail_mask == CLASSES[0]] = 1

        pc_mask[pc_mask != CLASSES[1]] = 0
        pc_mask[pc_mask == CLASSES[1]] = 1

        sc_mask[sc_mask != CLASSES[2]] = 0
        sc_mask[sc_mask == CLASSES[2]] = 1

        return np.stack((nail_mask, pc_mask, sc_mask), axis=2).astype(np.uint8) # .astype(np.float32)


    """
    def mask_multi_hot_encoding(self, mask_tensor):

        nail_mask = np.copy(mask_tensor)
        capillary_mask = np.copy(mask_tensor)

        nail_mask[nail_mask != CLASSES[0]] = 0
        nail_mask[nail_mask == CLASSES[0]] = 1

        capillary_mask[capillary_mask != CLASSES[1] and capillary_mask != CLASSES[2]] = 0
        capillary_mask[capillary_mask == CLASSES[1] or capillary_mask == CLASSES[2]] = 1

        return np.stack((nail_mask, capillary_mask), axis=2).astype(np.uint8)  # .astype(np.float32)
    """


def convrelu(in_channels, out_channels, kernel, padding):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        torch.nn.ReLU(inplace=True),
    )


class ResNetUNet(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = torch.nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = torch.nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = torch.nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, checkpoint_path, num_epochs=NUM_OF_EPOCHS):
    best_loss = 1e10
    metrics_lists = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #outputs = outputs.to(DEVICE)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              scheduler.step()
              for param_group in optimizer.param_groups:
                  print(f"LR: {param_group['lr']}")

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

            metrics_lists[phase].append({'loss': epoch_loss, 'dice': metrics['dice']/epoch_samples, 'bce': metrics['bce']/epoch_samples})

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model, metrics_lists


def save_preds_metrics(preds_dir, frames_masks_preds):
    preds_paths = []
    if os.path.exists(preds_dir) and os.path.isdir(preds_dir) and len(os.listdir(preds_dir)) > 0:
        dataset_utils.clean_dir(preds_dir)
    else:
        os.makedirs(preds_dir, exist_ok=True)

    frames, masks, preds = frames_masks_preds

    i = 0
    for frame in frames:
        cv2.imwrite(os.path.join(preds_dir, f'{str(i)}_frame.png'), frame)
        i += 1

    i = 0
    for mask in masks:
        cv2.imwrite(os.path.join(preds_dir, f'{str(i)}_mask.png'), mask)
        i += 1

    i = 0
    for pred in preds:
        pred_path = os.path.join(preds_dir, f'{str(i)}_pred.png')
        cv2.imwrite(pred_path, pred)
        preds_paths.append(pred_path)

        for j in range(pred.shape[2]):
            subpred = pred[:, :, j]
            subpred_path = os.path.join(preds_dir, f'{str(i)}_pred_{j+1}.png')
            cv2.imwrite(subpred_path, subpred)

        i += 1

    return preds_paths


def save_logs(logs_file, metrics_lists=None):
    saved = False
    if metrics_lists is not None:
        with open(logs_file, 'w') as f:
            f.write(f'{metrics_lists}')
            saved = True

    return saved


if __name__ == '__main__':
    dataset_utils.test_cuda()
    n_cpu = os.cpu_count()
    print(f'n_cpu: {n_cpu}')

    train_frames, val_frames, test_frames = get_train_val_test_frames(FRAMES_DIR)
    print(f'len(train_frames) = {len(train_frames)}, len(val_frames) = {len(val_frames)}, len(test_frames) = {len(test_frames)}')
    print(f'len(all_frames) = {len(train_frames.union(val_frames).union(test_frames))}')

    dataset_utils.clean_dir(MHE_MASKS_DIR)

    train_aug_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        list(train_frames),
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(None), # get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    val_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        list(val_frames),
        augmentation=None,
        preprocessing=get_preprocessing(None), # get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    image_datasets = {
        'train': train_aug_dataset, 'val': val_dataset
    }

    train_loader = DataLoader(train_aug_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=n_cpu//2)  # num_workers=8
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=n_cpu//2)  # num_workers=4

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    model = ResNetUNet(len(CLASSES)).to(DEVICE)
    save_summary(model)
    #del model
    model_path = make_model_dir(MODELS_DIR, type(model).__name__)
    print(f'model_path: {model_path}')
    best_model_path = os.path.join(model_path, BEST_MODEL)

    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)
    model, metrics_lists = train_model(model, optimizer_ft, exp_lr_scheduler, best_model_path, num_epochs=NUM_OF_EPOCHS)
    #print(f'metrics_lists: {metrics_lists}')
    saved = save_logs(os.path.join(model_path, 'train_val_logs.txt'), metrics_lists)
    print(f'save_logs(): {saved}')

    model.eval()  # Set model to the evaluation mode

    # Create a new simulation dataset for testing
    # create test dataset
    test_dataset = CapillariesSegDataset(
        FRAMES_DIR,
        MASKS_DIR,
        list(test_frames),
        augmentation=None,
        preprocessing=get_preprocessing(None),
        classes=CLASSES
    )

    test_dataloader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=n_cpu//2)

    # Get the first batch
    inputs, labels = next(iter(test_dataloader))
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    print(f'inputs.shape: {inputs.shape}')
    print(f'labels.shape: {labels.shape}')

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()#.numpy()
    print(f'pred.shape: {pred.shape}')

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
    #print(f'type(input_images_rgb): {type(input_images_rgb)}')
    input_labels_rgb = [reverse_transform(x) for x in labels.cpu()]
    #print(f'type(input_labels_rgb): {type(input_labels_rgb)}')
    input_preds_rgb = [reverse_transform(x) for x in pred]
    #print(f'type(input_preds_rgb): {type(input_preds_rgb)}')

    # Map each channel (i.e. class) to each color
    #target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
    #pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    preds_dir = os.path.join(model_path, 'preds')
    inputs_labels_pred = (input_images_rgb, input_labels_rgb, input_preds_rgb)
    preds_paths = save_preds_metrics(preds_dir, inputs_labels_pred)
    print(f'len(preds_paths) : {len(preds_paths)}')

    train_logs = {'bce': [], 'dice': [], 'loss': []}
    valid_logs = {'bce': [], 'dice': [], 'loss': []}

    for epoch in metrics_lists['train']:
        train_logs['loss'].append(epoch['loss'])
        train_logs['bce'].append(epoch['bce'])
        train_logs['dice'].append(epoch['dice'])

    for epoch in metrics_lists['val']:
        valid_logs['loss'].append(epoch['loss'])
        valid_logs['bce'].append(epoch['bce'])
        valid_logs['dice'].append(epoch['dice'])

    # Plot training & validation BCE loss values
    plt.figure(figsize=(45, 10))
    plt.subplot(131)
    plt.plot(train_logs['bce'])
    plt.plot(valid_logs['bce'])
    plt.title('Model BCE loss')
    plt.ylabel('BCE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation DICE loss values
    plt.subplot(132)
    plt.plot(train_logs['dice'])
    plt.plot(valid_logs['dice'])
    plt.title('Model DICE loss')
    plt.ylabel('DICE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(133)
    plt.plot(train_logs['loss'])
    plt.plot(valid_logs['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    train_val_plot_path = os.path.join(model_path, "train_val_plot.png")
    plt.savefig(train_val_plot_path, bbox_inches='tight')

    """
    unet = smp.Unet()
    pspnet = smp.PSPNet()
    fpn = smp.FPN()
    linknet = smp.Linknet()
    pan = smp.PAN()

    current_model = pan
    model_name = type(current_model).__name__

    metrics = [
        Accuracy(threshold=0.5, activation=None)  # activation='sigmoid' # activation='tanh')
    ]

    optimizer = torch.optim.Adam([
        dict(params=current_model.parameters(), lr=LR1)
    ])

    model_path = make_model_dir(MODELS_DIR, model_name, f'{NUM_OF_EPOCHS}_{ACTIVATION}_{type(LOSS).__name__}_{type(optimizer).__name__}_{type(metrics[0]).__name__}')
    print(f'model_path: {model_path}')
    best_model_path = os.path.join(model_path, BEST_MODEL)

    saved_augs = save_augs(DATA_DIR, train_aug_dataset)
    print(f'saved_augs: {saved_augs}')
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