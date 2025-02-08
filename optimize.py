from lib.datasets import MNIST
# from minst_model.dec_optimize import DEC
from lib.dec import DEC
from minst_model.idec import IDEC


# sdae_savepath = ("./model/mnist-csdae-189.pt")
# sdae_savepath = ("./DEC85/sdae-run-1.pt")
sdae_savepath = "./model/csdae-139.pt"

# finetune
mnist_train = MNIST('./dataset/mnist', train=True, download=False)
mnist_test = MNIST('./dataset/mnist', train=False)
X = mnist_train.train_data
y = mnist_train.train_labels
# X = mnist_test.test_data
# y = mnist_test.test_labels

dec = DEC(input_dim=784, z_dim=10, n_clusters=10,
          encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
print(dec)
dec.load_model(sdae_savepath)
dec.fit(X, y, lr=0.001, batch_size=256, num_epochs=500,
        update_interval=1)
dec_savepath = ("./model/optimize.pt")
dec.save_model(dec_savepath)