import torch
torch.manual_seed(1)

# Set Constant Values
batch_size = 512
class_map = {
    'PLANE': 0,
    'CAR': 1,
    'BIRD': 2,
    'CAT': 3,
    'DEER': 4,
    'DOG': 5,
    'FROG': 6,
    'HORSE': 7,
    'SHIP': 8,
    'TRUCK': 9
}

# Enable or disable visualizations
show_summary = True
show_dataset_analyze = True

from woolly.utils import get_device

# Check GPU availability
use_cuda, device = get_device()

# Load test and train loaders
from woolly.dataset import get_cifar_loader
# Get transforme functions
from woolly.transform import get_a_train_transform, get_a_test_transform

# create train and test loaders with transforms
train_loader, test_loader = get_cifar_loader(get_a_train_transform(), get_a_test_transform(), batch_size=batch_size, use_cuda=use_cuda)

from woolly.model import WyCifar10Net
from woolly.backpropagation import train, test, get_sgd_optimizer, get_crossentropy_criteria
from woolly.utils import initialize_weights, print_summary
from woolly.scheduler import one_cycle_lr_pt, one_cycle_lr_custom
from woolly.training import Training

# Set Hyper Parameters Train Params
epochs = 30
lr = 0.05
max_lr = 0.5
steps_per_epoch = len(train_loader)
dropout = True
drop_ratio = 0.1
lambda_l1 = 1e-7
momentum = 0.9
weight_decay = 0.000125
weight_decay = weight_decay/batch_size

print("Using Device:", device)
print("Epochs:", epochs)
print("Lr:", lr)
print("Max Lr:", max_lr)
print("Batch Size:", batch_size)
print("Dropout:", dropout)
print("Dropout Ratio:", drop_ratio)
print("Momentum:", momentum)
print("Weight Decay:", weight_decay)


# Here we will do following
# 1. Create instances of models
#   a. Model with BatchNormalization

norm='bn'
ctrain = train()

# Create model instance based on parameter which one to use
model = WyCifar10Net(ctype='depthwise_seperable', use1x1=True, base_channels=16, layers=1, drop_ratio=drop_ratio).apply(initialize_weights).to(device)
# Create optimizer instance based on hyper parameters
optimizer = get_sgd_optimizer(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
criteria = get_crossentropy_criteria(device)

# Create Pytorch One Cycle scheduler instance
pytorch_scheduler = one_cycle_lr_pt(
    optimizer, 
    lr=lr, 
    max_lr=max_lr, 
    steps_per_epoch=steps_per_epoch, 
    epochs=epochs, 
    anneal_strategy='linear'
)

# Create Custom One Cycle schedule instance
custom_scheduler = one_cycle_lr_custom(
    optimizer, 
    lr=lr, 
    max_lr=max_lr, 
    steps_per_epoch=steps_per_epoch, 
    epochs=epochs, 
    anneal_strategy='linear'
)

# Create instance of trainer with all params
trainer = Training(
    model,
    optimizer,
    criteria,
    custom_scheduler,
    ctrain,
    test,
    train_loader,
    test_loader,
    lr,
    epochs,
    device,
    dropout
)

if show_summary:
    print_summary(model, input_size=(3, 32, 32))

# Run trainer
trainer.run()