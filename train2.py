import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from PIL import Image


triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
CUDA_LAUNCH_BLOCKING = 1


def get_device():
    device = ('cuda'
              if torch.cuda.is_available()
              else 'mps'
              if torch.backends.mps.is_available()
              else 'cpu'
              )
    return torch.device(device)


def reverse_transform(input):
    # if len(input.shape) == 3:
    #     input = input.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    # else:
    #     mean = np.array([0.5])
    #     std = np.array([0.25])

    # input = std*input+mean
    input = np.clip(input, 0, 1)
    input = (input*255).astype(np.uint8)

    return input


class LocalDataset(Dataset):
    def __init__(self, path_image, transform=None, start=0, count=0):
        self.transform = transform

        self._image_dir = path_image
        self._start = start
        self._count = count

        image_names_list: list[str] = os.listdir(self._image_dir)
        self._image_files: list[str] = [
            file for file in image_names_list if file.endswith('.jpg')]
        
        
        # self.input_images = []
        # for i in range(start, start+count):
        #     self.input_images.append(np.asarray(
        #         Image.open(image_dir+"/"+image_files[i]).convert('L')))

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        anchor = np.asarray(Image.open(
            self._image_dir+"/"+self._image_files[idx]).convert('RGB'))
        # print(anchor.shape)
        prefix = self._image_files[idx][0:7]
        if self._image_files[idx+1][0:7] == prefix:
            positive = np.asarray(Image.open(
                self._image_dir+"/"+self._image_files[idx+1]).convert('RGB'))
        else:
            positive = np.asarray(Image.open(
                self._image_dir+"/"+self._image_files[idx-1]).convert('RGB'))

        rand_idx = random.randint(self._start, self._count+self._start-1)
        while (self._image_files[rand_idx][0:7] == prefix):
            rand_idx = random.randint(self._start, self._count+self._start-1)

        negative = np.asarray(Image.open(
            self._image_dir+"/"+self._image_files[rand_idx]).convert('RGB'))

        # anchor = anchor.transpose(2, 0, 1)
        # positive = positive.transpose(2, 0, 1)
        # negative = negative.transpose(2, 0, 1)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return [anchor, positive, negative]


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.25])
    ])


def get_train_data_loader(path_image, batch_size, start, count):
    transforms = get_transforms()
    train_set = LocalDataset(path_image, transforms, start, count)
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_dataloader


def get_validation_data_loader(path_image, batch_size, start, count):
    transforms = get_transforms()
    validation_set = LocalDataset(
        path_image,  transforms, start, count)
    validation_dataloader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return validation_dataloader


def get_test_data_loader(path_image, batch_size, start, count):
    transforms = get_transforms()
    test_set = LocalDataset(path_image, transforms, start, count)
    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_dataloader


def calc_loss(anchor, positive, negative, metrics):

    loss = triplet_loss(anchor, positive, negative)

    metrics['loss'] += loss.data.cpu().numpy()*anchor.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k]/epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def training(model: torch.nn.Module, optimizer, scheduler, dataloaders, num_epochs=100):
    device = get_device()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e20

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        since = time.time()
        train_loss = train_model(
            model, scheduler, optimizer, dataloaders['train'], device)
        validation_loss = validate_model(
            model, optimizer, dataloaders['val'], device)
        if validation_loss < best_loss:
            print('model improved')
            best_loss = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "./Models/m5.pt")
            with open('./Models/m5.txt', 'w') as f:
                f.write(str(epoch))
                f.write('\n')
                f.write(str(best_loss))
                f.write('\n')
                for param_group in optimizer.param_groups:
                    f.write(str(param_group['lr']))
                    f.write('\n')
        # if epoch % 5 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': epoch_loss,
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'dataloaders_state_dict': dataloaders,
        #     }, f'Training all models/genAndReal2_{epoch}')
        with open('./Losses/l5.txt', 'a') as f1:
            f1.write(str(train_loss))
            f1.write(' ')
            f1.write(str(validation_loss))
            f1.write('\n')
        time_elapsed = time.time()-since
        print(f'{time_elapsed} seconds')

    print(f'Best validation loss: {best_loss}')
    model.load_state_dict(best_model)
    return model


def train_model(model, scheduler, optimizer, dataloader, device):
    model.train()
    scheduler.step()
    for param_group in optimizer.param_groups:
        print('LR', param_group['lr'])
    metrics = defaultdict(float)
    epoch_samples = 0
    sample_index = 0

    for anchors, positives, negatives in dataloader:
        # !!!!!!!!!!DODATO JER NEKAD CUDA OTKAZE IDK STO I KAD I KAKO!!!!!!!!!!!!!!!!!!!!!!!!!
        device = get_device()
        if (sample_index == 125):
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "./TempModel/m5.pt")
            with open('./TempModel/l5.txt', 'a') as f1:
                f1.write(str(loss))
                f1.write(' ')
            sample_index = 0
        else:
            sample_index += 1  # ISTO DODATO KAO BACKUP

        anchors = anchors.to(device)
        size = anchors.size(0)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):

            outputs_anchors = model(anchors)
            del anchors
            # torch.cuda.empty_cache()

            positives = positives.to(device)
            outputs_positives = model(positives)
            del positives
            # torch.cuda.empty_cache()

            negatives = negatives.to(device)
            outputs_negatives = model(negatives)
            del negatives
            # torch.cuda.empty_cache()

            loss = calc_loss(outputs_anchors, outputs_positives,
                             outputs_negatives, metrics)
            loss.backward()
            optimizer.step()

        epoch_samples += size

    print_metrics(metrics, epoch_samples, 'train')
    return metrics['loss']/epoch_samples


def validate_model(model, optimizer, dataloader, device):
    model.eval()
    metrics = defaultdict(float)
    epoch_samples = 0
    for anchors, positives, negatives in dataloader:
        anchors = anchors.to(device)
        size = anchors.size(0)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):

            outputs_anchors = model(anchors)
            del anchors
            # torch.cuda.empty_cache()

            positives = positives.to(device)
            outputs_positives = model(positives)
            del positives
            # torch.cuda.empty_cache()

            negatives = negatives.to(device)
            outputs_negatives = model(negatives)
            del negatives
            # torch.cuda.empty_cache()

            _ = calc_loss(outputs_anchors, outputs_positives,
                          outputs_negatives, metrics)

        epoch_samples += size

    print_metrics(metrics, epoch_samples, 'val')
    epoch_loss = metrics['loss']/epoch_samples

    return epoch_loss


def test_model(model, dataloader: DataLoader, device, vizualize=False):
    model.eval()
    metrics = defaultdict(float)
    loss = 0
    with torch.no_grad():
        for anchors, positives, negatives in dataloader:
            device = get_device()

            anchors = anchors.to(device)
            outputs_anchors = model(anchors)
            del anchors

            positives = positives.to(device)
            outputs_positives = model(positives)
            del positives

            negatives = negatives.to(device)
            outputs_negatives = model(negatives)
            del negatives

            loss += calc_loss(outputs_anchors, outputs_positives,
                              outputs_negatives, metrics)

            if vizualize:
                vizualize_prediction(
                    outputs_anchors, outputs_positives, outputs_negatives)
    loss /= len(dataloader)
    print(f'Avg loss: {loss}')


def vizualize_prediction(outputs_anchors: torch.Tensor, outputs_positives: torch.Tensor, outputs_negatives: torch.Tensor):
    return
    # prediction_probs = torch.sigmoid(prediction)
    # input_img = reverse_transform((input[0][0]).data.cpu().numpy())
    # label_img = reverse_transform((label[0][0]).data.cpu().numpy())
    # prediction_img = reverse_transform(
    #     (prediction_probs[0][0]).data.cpu().numpy())
    # prediction_img[prediction_img > 127] = 255
    # prediction_img[prediction_img <= 127] = 0
    # fig, axeslist = plt.subplots(1, 3)
    # axeslist.ravel()[0].imshow(input_img, cmap='gray')
    # axeslist.ravel()[1].imshow(label_img, cmap='gray')
    # axeslist.ravel()[2].imshow(prediction_img, cmap='gray')
    # plt.show()


def run(FaceNet, model_path=""):
    train_index = 0
    train_count = 95897

    validation_index = 95897
    validation_count = 4959

    # test_index =
    # test_count =    # TODO:: inicijalizacija

    batch_size = 4
    num_epoches = 200
    lr = 5e-4

    device = get_device()
    # model = FaceNet(4096).to(device)
    model = FaceNet(1024).to(device)

    if model_path != "":
        print("Loading model from path")
        model.load_state_dict(torch.load(model_path))

    print(model)

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        # weight_decay=1e-3 #Maybe
    )

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer=optimizer_ft,
        step_size=5,
        gamma=0.5
    )

    train_dataloader = get_train_data_loader(
        '../../DS1/CASIA-WebFace_crop', batch_size, train_index, train_count)
    validation_dataloader = get_validation_data_loader(
        '../../DS1/CASIA-WebFace_crop', batch_size, validation_index, validation_count)

    dataloaders = {
        'train': train_dataloader,
        'val': validation_dataloader
    }

    model = training(model, optimizer_ft, exp_lr_scheduler,
                     dataloaders, num_epochs=num_epoches)

    # torch.save(model.state_dict(), "./ModelsReal/model01.pt")

    # test_dataloader = get_test_data_loader(
    #     '../nucelus-image-gen/ImageGen/Mixed/Test img', '../nucelus-image-gen/ImageGen/Mixed/Test mask', batch_size, test_index, test_count)
    # test_model(model, test_dataloader, device)


def load_test_model(FaceNet, index, count):
    vector_length = 1024
    batch_size = 10
    model_path = './Save/m5.pt'
    dataset_image_path = '../../DS1/CASIA-WebFace_crop'
    device = get_device()
    model = FaceNet(vector_length).to(device)
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])

    test_dataloader = get_test_data_loader(
        dataset_image_path, batch_size, index, count)
    test_model(model, test_dataloader, device, vizualize=False)
