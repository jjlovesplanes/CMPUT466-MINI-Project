import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNETmodel import UNET
from CNNmodel import CNN
from DeepLabV3model import DeepLabV3
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 4
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "carvana-image-masking-challenge/train_img/"
TRAIN_MASK_DIR = "carvana-image-masking-challenge/train_masks/"
VAL_IMG_DIR = "carvana-image-masking-challenge/valid_img/"
VAL_MASK_DIR = "carvana-image-masking-challenge/valid_masks/"
TEST_IMG_DIR = "carvana-image-masking-challenge/test_img/"
TEST_MASK_DIR = "carvana-image-masking-challenge/test_masks/"

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

test_loader, _ = get_loaders(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        None,  # Assuming you don't have separate validation set for testing
        None,
        BATCH_SIZE,
        train_transform,
        None,  # No need for validation transforms during testing
        NUM_WORKERS,
        PIN_MEMORY,
    )

######################################################################################
def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # print("predictions.shape is ", predictions.shape)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


######################################################################################
def trainAndValidateUNET():
    UNETmodel = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(UNETmodel.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_UNET_checkpoint.pth.tar"), UNETmodel)


    check_accuracy(val_loader, UNETmodel, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, UNETmodel, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": UNETmodel.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "my_UNET_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, UNETmodel, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, UNETmodel, folder="saved_images_UNET/", device=DEVICE
        )

######################################################################################
def trainAndValidateCCN():
    CNNmodel = CNN(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(CNNmodel.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_CNN_checkpoint.pth.tar"), CNNmodel)


    check_accuracy(val_loader, CNNmodel, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, CNNmodel, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": CNNmodel.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "my_CNN_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, CNNmodel, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, CNNmodel, folder="saved_images_CNN/", device=DEVICE
        )

######################################################################################
def trainAndValidateDeepLabV3():
    DeepLabV3model = DeepLabV3(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(DeepLabV3model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_DeepLabV3_checkpoint.pth.tar"), DeepLabV3model)


    check_accuracy(val_loader, DeepLabV3model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, DeepLabV3model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": DeepLabV3model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "my_DeepLabV3_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, DeepLabV3model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, DeepLabV3model, folder="saved_images_DeepLabV3/", device=DEVICE
        )

######################################################################################
def test_CNN():
    CNNmodel = CNN(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(CNNmodel.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Load the checkpoint
    checkpoint = torch.load("my_CNN_checkpoint.pth.tar")
    CNNmodel.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Set model to evaluation mode
    CNNmodel.eval()

    # Check accuracy on the test set
    check_accuracy(test_loader, CNNmodel, device=DEVICE)

    # Save predictions as images
    save_predictions_as_imgs(
        test_loader, CNNmodel, folder="saved_images_TEST_CNN/", device=DEVICE
    )

######################################################################################
def test_UNET():
    UNETmodel = UNET(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(UNETmodel.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Load the checkpoint
    checkpoint = torch.load("my_UNET_checkpoint.pth.tar")
    UNETmodel.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Set model to evaluation mode
    UNETmodel.eval()

    # Check accuracy on the test set
    check_accuracy(test_loader, UNETmodel, device=DEVICE)

    # Save predictions as images
    save_predictions_as_imgs(
        test_loader, UNETmodel, folder="saved_images_TEST_UNET/", device=DEVICE
    )
######################################################################################
def test_DeepLabV3():
    DeepLabV3model = DeepLabV3(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(DeepLabV3model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Load the checkpoint
    checkpoint = torch.load("my_DeepLabV3_checkpoint.pth.tar")
    DeepLabV3model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Set model to evaluation mode
    DeepLabV3model.eval()

    # Check accuracy on the test set
    check_accuracy(test_loader, DeepLabV3model, device=DEVICE)

    # Save predictions as images
    save_predictions_as_imgs(
        test_loader, DeepLabV3model, folder="saved_images_TEST_DeepLabV3/", device=DEVICE
    )






if __name__ == "__main__":
    print("\n\nWe are now training and validating CNN")
    trainAndValidateCCN()

    print("\n\nWe are now training and validating UNET")
    trainAndValidateUNET()
    
    print("\n\nWe are now training and validating DeepLabV3")
    trainAndValidateDeepLabV3()
    
    print("\n\nWe are now testing CNN")
    test_CNN()

    print("\n\nWe are now testing UNET")
    test_UNET()

    print("\n\nWe are now testing DeepLabV3")
    test_DeepLabV3()


