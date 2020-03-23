import uuid

from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torchvision import datasets,transforms,models
import imgaug as ia
import imgaug.augmenters as iaa
import os


def load_dataset(image_folder, input_size, train_per= 0.7, batch_size= 8) -> ():
    transform=transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    full_data=datasets.ImageFolder(image_folder, transform=transform)
    train_size=int(train_per*len(full_data))
    val_size=len(full_data)-train_size
    print(train_size, val_size)
    train_set,val_set=torch.utils.data.random_split(full_data,[train_size,val_size])
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size)
    val_loader=torch.utils.data.DataLoader(val_set,batch_size=batch_size)
    return  train_loader, val_loader


def train_model(model: nn.Module,train_loader: data.DataLoader, val_loader: data.DataLoader, n_epochs: int = 100):
    # check if the gpu is available
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #criterion=nn.NLLLoss()
    #optimizer=optim.Adam(model.classifier.parameters(),lr=0.003)
    criterion=nn.CrossEntropyLoss()
    # specify optimizer (stochastic gradient descent) and learning rate = 0.001
    optimizer=optim.SGD(model.classifier.parameters(),lr=0.001, momentum=0.9)
    #optimizer=optim.SGD(model.classifier.parameters(),lr=0.000075)
    model.to(device)
    # number of epochs to train the model
    valid_loss_min=np.Inf  # track change in validation loss
    for epoch in range(1,n_epochs+1):
        # keep track of training and validation loss
        train_loss=0.0
        valid_loss=0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data,target in train_loader:
            # move tensors to GPU or the CPU
            data,target=data.to(device),target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output=model(data)
            # calculate the batch loss
            loss=criterion(output,target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss+=loss.item()

        ######################
        # validate the model #
        ######################
        model.eval()
        accuracy=0
        for data,target in val_loader:
            # move tensors to GPU if CUDA is available
            data,target=data.to(device),target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            logps=model(data)
            # calculate the batch loss
            loss=criterion(logps,target)
            # update average validation loss
            valid_loss+=loss.item()
            # Calculate accuracy
            ps=torch.exp(logps)
            top_p,top_class=ps.topk(1,dim=1)
            equals=top_class == target.view(*top_class.shape)
            accuracy+=torch.mean(equals.type(torch.float)).item()

        # calculate average losses
        train_loss=train_loss/len(train_loader)
        valid_loss=valid_loss/len(val_loader)
        accuracy=accuracy/len(val_loader)
        print(f"Epoch {epoch+1}/{epoch}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Test loss: {valid_loss:.3f}.. "
              f"Test accuracy: {accuracy:.3f}")
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(),'convmodel.pt')
            valid_loss_min=valid_loss


def augment_data(images, out_folder):
    ia.seed(1)
    seq=iaa.Sequential([
        iaa.Crop(px=(0,16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0,3.0)),  # blur images with a sigma of 0 to 3.0
        iaa.Sharpen(alpha=(0,1.0),lightness=(0.75,1.5)),
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Same as sharpen, but for an embossing effect.
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    ])
    images_aug=seq(images=images)
    os.makedirs(out_folder,exist_ok=True)
    for img in images_aug:
        img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        cv2.imwrite(os.path.join(out_folder,"{}.jpg".format(uuid.uuid4())),img)

def create_model():
    # create model
    model = models.vgg16(pretrained=True)
    #print(model.classifier)
    for param in model.features.parameters():
        param.requires_grad=False
    n_inputs=model.classifier[6].in_features
    model.classifier[6]=nn.Linear(n_inputs,2)
    return model

if __name__ == '__main__':
    model = create_model()
    train_loader, val_loader = load_dataset(image_folder="./data/train", input_size=224, train_per=0.7, batch_size=12)
    train_model(model, train_loader=train_loader, val_loader=val_loader, n_epochs= 20)

