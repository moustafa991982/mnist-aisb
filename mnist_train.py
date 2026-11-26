# %% [markdown]
# # MNIST training loop (AISB prerequisite exercise)
# Run each cell with "Run Cell" (or Shift+Enter) in the VS Code Python Interactive Window.

# %% Imports and device
# ---------------------------------------------------------------------------
# Load the libraries we need. Nothing is printed by the imports themselves;
# if one is missing you get "ModuleNotFoundError".
# ---------------------------------------------------------------------------
import torch                                   # tensors (GPU-capable arrays) + automatic gradients
from torch import nn                           # layers, activations, loss functions, nn.Module base class
from torch.utils.data import DataLoader        # serves a dataset in shuffled mini-batches
from torchvision import datasets               # ready-made datasets such as MNIST
from torchvision.transforms import ToTensor    # converts an image (pixels 0-255) to a float tensor (0.0-1.0)
import os                                      # file-system helpers (used below to show where data is saved)


# Pick the fastest hardware available: NVIDIA GPU ("cuda"), Apple GPU ("mps"), else the CPU.
# The model and every batch of data must later be moved to this same device with .to(device).
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# OUTPUT: e.g. "Using device: cuda" or "Using device: cpu"
print(f"Using device: {device}")

# %% Data: download MNIST (60k training, 10k test images of handwritten digits)
# ---------------------------------------------------------------------------
# Create the two dataset objects. "data" is just a folder name (a string, not a variable);
# torchvision stores the files in <current folder>/data/MNIST/raw and downloads them
# the first time (you may see download progress messages). Later runs reuse the files.
#   train=True  -> the 60,000 training images
#   train=False -> the 10,000 test images (never trained on; used to measure accuracy)
#   transform=ToTensor() -> each image becomes a tensor of shape [1, 28, 28], values 0.0-1.0
# ---------------------------------------------------------------------------
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# OUTPUT: the working directory "data" is relative to, e.g. /home/hp/mnist-aisb
print(os.getcwd())
# OUTPUT: the full path "data" resolves to, e.g. /home/hp/mnist-aisb/data
print(os.path.abspath("data"))

# How many images the model sees at once before each weight update.
batch_size = 64
# Training loader: shuffle=True gives a new random order every epoch (helps training).
# 60,000 images / 64 per batch -> 938 batches per epoch (the last batch is smaller).
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# Test loader: no shuffling needed, we only measure accuracy. 10,000 / 64 -> 157 batches.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Grab ONE batch just to inspect it.
#   X = images, shape [64, 1, 28, 28] = [batch, colour channels (1 = grayscale), height, width]
#   y = labels, shape [64]            = one digit 0-9 per image
X, y = next(iter(train_dataloader))
# OUTPUT: Batch shape: torch.Size([64, 1, 28, 28]) labels shape: torch.Size([64])
print("Batch shape:", X.shape, "labels shape:", y.shape)  # [64, 1, 28, 28], [64]

# %% Model: a small fully connected network (subclass of nn.Module)
# ---------------------------------------------------------------------------
# NOTE: naming this class "X" REPLACES the batch tensor X from the cell above
# (same name, last assignment wins). Fine as long as you don't re-run the Data cell
# afterwards; if you do, X becomes a tensor again and "X()" in the last cell fails
# with "TypeError: 'Tensor' object is not callable".
# ---------------------------------------------------------------------------
class X(nn.Module):
    def __init__(self):
        super().__init__()                       # required: sets up nn.Module internals
        self.flatten = nn.Flatten()              # [64, 1, 28, 28] -> [64, 784] (each image becomes one row of 784 pixels)
        self.linear_relu_stack = nn.Sequential(  # layers applied one after another:
            nn.Linear(28 * 28, 512),             #   784 pixels -> 512 numbers (weights: 784*512 + 512 biases)
            nn.ReLU(),                           #   non-linearity: negative values -> 0
            nn.Linear(512, 512),                 #   512 -> 512
            nn.ReLU(),
            nn.Linear(512, 10),  # 10 output scores (logits), one per digit
        )

    # forward() defines what happens when you call model(x).
    # Input [batch, 1, 28, 28] -> output [batch, 10] raw scores (higher = more likely that digit).
    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))

    # The predicted digit = position of the highest of the 10 scores. Output shape [batch].
    def predict(self, x):
        return self(x).argmax(dim=1)

    # Loss for one batch: how wrong the scores are compared to the true labels y.
    def loss(self, x, y, loss_fn):
        return loss_fn(self(x), y)

    # Accuracy and average loss over a whole dataloader (similar to test_loop below,
    # but returns the numbers instead of printing them). Returns (accuracy, average_loss).
    def evaluate(self, dataloader, loss_fn):
        self.eval()                                  # evaluation mode
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():                        # no gradients needed -> faster, less memory
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = self(X_batch)                                    # [batch, 10] scores
                total_loss += loss_fn(logits, y_batch).item()             # .item(): 1-element tensor -> Python float
                correct += (self.predict(X_batch) == y_batch).sum().item()  # count right answers in this batch
                total += y_batch.size(0)                                  # number of images in this batch
        average_loss = total_loss / max(len(dataloader), 1)   # mean loss per batch (max(...,1) avoids dividing by 0)
        accuracy = correct / max(total, 1)                     # fraction correct, 0.0-1.0
        return accuracy, average_loss


# A second name for the same class, so "NeuralNetwork()" and "X()" both work.
NeuralNetwork = X
# Create the model (weights start random) and move its weights to the chosen device.
model = NeuralNetwork().to(device)
# OUTPUT: a summary of the layers, e.g.
#   X(
#     (flatten): Flatten(start_dim=1, end_dim=-1)
#     (linear_relu_stack): Sequential(
#       (0): Linear(in_features=784, out_features=512, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=512, out_features=512, bias=True)
#       (3): ReLU()
#       (4): Linear(in_features=512, out_features=10, bias=True)
#     )
#   )
print(model)

# %% Hyperparameters, loss function and optimizer
# ---------------------------------------------------------------------------
# Nothing is printed in this cell; it only sets things up.
# ---------------------------------------------------------------------------
learning_rate = 0.1   # step size: how far the weights move on each update
epochs = 3            # number of full passes over the 60,000 training images

# CrossEntropyLoss: takes the 10 raw scores + the true digit and returns one number
# (lower = better). Before training it is about 2.3 (= random guessing among 10 classes).
loss_fn = nn.CrossEntropyLoss()
# SGD (stochastic gradient descent): after each batch, weight = weight - lr * gradient.
# model.parameters() tells it which tensors it is allowed to update.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% Training and test loops
# ---------------------------------------------------------------------------
# This cell only DEFINES two functions; nothing runs or prints until the Train cell calls them.
# (The X and y inside these functions are local names; they don't touch the class X.)
# ---------------------------------------------------------------------------
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()                                    # training mode
    for batch, (X, y) in enumerate(dataloader):      # batch = 0, 1, 2, ... 937
        X, y = X.to(device), y.to(device)            # move this batch to the same device as the model

        pred = model(X)              # 1. forward pass
        loss = loss_fn(pred, y)      # 2. compute loss

        optimizer.zero_grad()        # 3. clear old gradients
        loss.backward()              # 4. backpropagation: fills .grad on each parameter
        optimizer.step()             # 5. update the weights
        # Print progress every 200 batches (batches 0, 200, 400, 600, 800 -> 5 lines per epoch).
        # OUTPUT: e.g. "  loss: 2.301245  [    0/60000]" ... "  loss: 0.153201  [51200/60000]"
        #   left number  = loss on the current batch (should go DOWN over time)
        #   [seen/total] = how many training images have been processed so far this epoch
        if batch % 200 == 0:
            print(f"  loss: {loss.item():>7f}  [{batch * len(X):>5d}/{len(dataloader.dataset):>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()                     # evaluation mode
    test_loss, correct = 0.0, 0
    with torch.no_grad():  # no gradients needed for evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)                                                  # [batch, 10] scores
            test_loss += loss_fn(pred, y).item()                             # add this batch's loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correct predictions
    test_loss /= len(dataloader)                   # average loss per batch (157 batches)
    accuracy = correct / len(dataloader.dataset)   # correct out of 10,000 test images
    # OUTPUT: e.g. "Test: accuracy 96.5%, avg loss 0.1123"
    print(f"Test: accuracy {100 * accuracy:.1f}%, avg loss {test_loss:.4f}")
    return accuracy

# %% Train
# ---------------------------------------------------------------------------
# The actual training. Each epoch = one pass over all 60,000 training images,
# then a check on the 10,000 test images. Takes seconds on a GPU, a minute or so on a CPU.
# OUTPUT per epoch, roughly:
#   Epoch 1
#     loss: 2.30...  [    0/60000]      <- starts near 2.3 (random guessing)
#     loss: 0.4...   [12800/60000]
#     ... (5 loss lines)
#   Test: accuracy ~95%, avg loss ~0.15
# Accuracy should rise and loss should fall from epoch to epoch (exact numbers vary run to run).
# ---------------------------------------------------------------------------
for t in range(epochs):              # t = 0, 1, 2
    print(f"Epoch {t + 1}")          # OUTPUT: "Epoch 1", "Epoch 2", "Epoch 3"
    train_loop(train_dataloader, model, loss_fn, optimizer)   # update weights over all training batches
    test_loop(test_dataloader, model, loss_fn)                # measure accuracy on unseen test images
print("Done!")                       # OUTPUT: "Done!"

# %% Save and reload the model
# ---------------------------------------------------------------------------
# Save only the learned weights (the "state_dict": a dict of layer name -> weight tensor)
# to the file mnist_model.pth in the current folder. Nothing is printed.
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "mnist_model.pth")

# Build a FRESH model (random weights), then load the saved weights into it.
# weights_only=True = only load tensors from the file (safer than loading arbitrary Python objects).
reloaded = X().to(device)
reloaded.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
# OUTPUT: the same accuracy as the last epoch above, which proves the save/load worked,
# e.g. "Test: accuracy 97.0%, avg loss 0.0950"
test_loop(test_dataloader, reloaded, loss_fn)

# %%
