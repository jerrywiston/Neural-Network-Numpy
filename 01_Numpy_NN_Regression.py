import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork

def Visualization(x_train, y_train, y_gt, y_pred):
    plt.figure()
    plt.plot(x_train, y_train,'b.')
    plt.plot(x_train, y_gt, 'g.')
    plt.plot(x_train, y_pred,'r.')

if __name__ == "__main__":
    # Data Simulation
    data_size = 1000
    x_train = np.linspace(-5, 5, data_size).reshape(-1,1)
    y_gt = np.power(x_train, 2)
    y_train = y_gt + np.random.randn(y_gt.shape[0], y_gt.shape[1])

    # Create NN Model
    nn = NeuralNetwork.NN(1,32,1,"linear")
    
    # Show Initial Curve
    nn.feed({"x":x_train})
    y_pred = nn.forward()
    Visualization(x_train, y_train, y_gt, y_pred)

    # Training the Model
    loss_rec = []
    batch_size = 128
    for i in range(10001):
        # Sample Data Batch
        batch_id = np.random.choice(data_size, batch_size)
        x_batch = x_train[batch_id]
        y_batch = y_train[batch_id]

        # Forward & Backward & Update
        nn.feed({"x":x_batch, "y":y_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-3)

        # Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)
        if i%100 == 0:
            print("[Iteration {:5d}] Loss={:.4f}".format(i,loss))

    # Show Fitting Curve
    nn.feed({"x":x_train})
    y_pred = nn.forward()
    Visualization(x_train, y_train, y_gt, y_pred)

    # Show Loss Record
    plt.figure()
    plt.plot(loss_rec[100:])
    plt.show()
    