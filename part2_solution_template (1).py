# Don't erase the template code, except "Your code here" comments.

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder as IF
from torchvision import transforms
import torchvision.models as models
import tqdm
# Your code here...

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    

    if kind == 'train':
        dataset = IF(root = path+'/'+kind+'/', transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        
        return DataLoader(dataset, batch_size=64)
    elif kind == 'val':
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        dataset = IF(root = path+'/'+kind+'/', transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        return DataLoader(dataset, num_workers = 0, batch_size=64)

    else:
        print('Please. specify the type of images')
    # Your code here

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    
    resnet18 = models.resnet18(pretrained = False)
    # since pretrained models are in restrictions
    # we set pretrained = False
    resnet18.fc = torch.nn.Linear(512, 200, bias = True)
    return resnet18

    # Your code here

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """

    return torch.optim.Adam(model.parameters(), lr = 1e-4)

    # Your code here

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    out = model(batch)
    return out



    # Your code here

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    history_loss = []
    history_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, target = data
            inputs.to(device)
            target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    history_acc.apppend(correct / total)
    return accuracy
            
    # Your code here

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    epochs = 50
    criterion = torch.nn.CrossEntropyLoss()
    history_loss = []
    accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in tqdm.notebook.tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            correct += (preds == labels).float().sum()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))

                print(f'accuracy is :{100 * correct / 100000}')
                running_loss = 0.0
        history_loss.append(running_loss)
        history_acc.append(correct)
        



    # Your code here

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"
    
    return md5_checksum, google_drive_link
