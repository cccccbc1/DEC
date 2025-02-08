import torch.utils.data
from stackedDAE import StackedDAE
from dec import DEC
# from dec_optimize import DEC
from idec import IDEC
from dec_pytorch.lib.datasets import MNIST


# finetune
# sdae_savepath = "./train_model/0.1_815-876.pt"
sdae_savepath = "./train_model/model_epoch5000.pt"
# sdae_savepath = "./mnist-csdae-189.pt"

mnist_train = MNIST('../dataset/mnist', train=True, download=False)
mnist_test = MNIST('../dataset/mnist', train=False)
X = mnist_train.train_data
y = mnist_train.train_labels
dec = DEC(input_dim=784, z_dim=10, n_clusters=10,
   encodeLayer=[500, 500, 2000], activation="relu", dropout=0)

print(dec)
dec.load_model(sdae_savepath)
dec.fit(X, y, lr=0.001, batch_size=256, num_epochs=500,
    update_interval=1)
dec_savepath = ("./model/sdae.pt")
dec.save_model(dec_savepath)