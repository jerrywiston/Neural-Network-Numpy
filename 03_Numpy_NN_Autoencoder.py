import numpy as np
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def PlotFig(samples, fig_size, samp_size):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    gs = gridspec.GridSpec(fig_size[0], fig_size[1])
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(samp_size), cmap='gray')
    return fig

def SaveFig(fname, samples, fig_size, samp_size):
    fig = PlotFig(samples, fig_size, samp_size)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_train = x_train.astype(np.float32) / 255.

    # Create NN Model
    nn = NeuralNetwork.NN(784,128,784,"sigmoid")

    # Training the Model
    out_folder = 'out/'
    import os
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    loss_rec = []
    batch_size = 16
    for i in range(10001):
        # Sample Data Batch
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]
        corrupt = np.random.choice(2, x_batch.shape, p=[0.9, 0.1]).astype(np.float32)
        x_batch_noise = np.maximum(x_batch, corrupt)
        
        # Forward -> Backward -> Update
        nn.feed({"x":x_batch_noise, "y":x_batch})
        x_re = nn.forward()
        nn.backward()
        nn.update(1e-3)

        # Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)

        if i%100 == 0:
            print("[Iteration {:5d}] Loss={:.4f}".format(i,loss))

        if i%1000 == 0:
            print("Save Fig ...")
            x_fig = np.concatenate((x_batch[0:4], x_re[0:4], x_batch_noise[4:8], x_re[4:8]), axis=0)
            samp_name = out_folder + str(int(i/1000)).zfill(4) + '.png'
            SaveFig(samp_name, x_fig, [4,4], [28,28])
    
    plt.title("Loss")
    plt.plot(loss_rec)
    plt.show()
