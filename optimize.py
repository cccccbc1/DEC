# from lib.dec import DEC
from lib.dec_optimize import DEC
from lib.datasets import MNIST

sdae_savepath = ("./model/sdae-run-1.pt")

# finetune
mnist_train = MNIST('./dataset/mnist', train=True, download=True)
mnist_test = MNIST('./dataset/mnist', train=False)
X = mnist_train.train_data
y = mnist_train.train_labels

dec = DEC(input_dim=784, z_dim=10, n_clusters=10,
          encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
print(dec)
dec.load_model(sdae_savepath)
dec.fit(X, y, lr=0.01, batch_size=256, num_epochs=100,
        update_interval=1)
dec_savepath = ("./model/optimize.pt")
dec.save_model(dec_savepath)