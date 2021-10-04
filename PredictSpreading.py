import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATASET_SET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_ENTRIES_SET = [10000, 50000, 200000]

for DATASET in DATASET_SET:
    for NUM_ENTRIES in NUM_ENTRIES_SET:
        print("FOR DATASET: " + str(DATASET) + "; NUM_ENTRIES: " + str(NUM_ENTRIES))
        print()

        def load_ds(path):
            raw_ds = pd.read_csv(path)
            raw_ds = raw_ds.to_numpy()

            X = raw_ds[:NUM_ENTRIES, 1:-1]
            y = raw_ds[:NUM_ENTRIES, -1:]
            n = len(X)
            a, b = int(n*0.7), int(n*0.15)
            xtrain, ytrain = X[:a], y[:a]
            xval, yval = X[a:a+b], y[a:a+b]
            xtest, ytest = X[a+b:], y[a+b:]

            return xtrain, ytrain, xval, yval, xtest, ytest


        LOG_OUTPUTS = True
        n_features = 6

        path = "dataset/datapoints-" + str(DATASET) + ".csv"
        xtrain_indexed, ytrain, xval_indexed, yval, xtest_indexed, ytest = load_ds(path)


        xtrain_indexed = torch.from_numpy(xtrain_indexed).float()
        ytrain = torch.from_numpy(np.array(ytrain)).float()

        xval_indexed = torch.from_numpy(xval_indexed).float()
        yval = torch.from_numpy(np.array(yval)).float()

        xtest_indexed = torch.from_numpy(xtest_indexed).float()
        ytest = torch.from_numpy(np.array(ytest)).float()


        if LOG_OUTPUTS:
            ytrain = torch.log(ytrain)
            yval = torch.log(yval)
            ytest = torch.log(ytest)


        # define the network configuration
        class Net(nn.Module):
            def __init__(self, layers_configuration, drop_prob):
                super(Net, self).__init__()
                self.layers = len(layers_configuration)

                self.linears = nn.ModuleList()
                for i in range(self.layers - 1):
                    self.linears.append(
                        nn.Linear(layers_configuration[i], layers_configuration[i+1]))

                self.dropout = nn.Dropout(p=drop_prob)

            def forward(self, x):
                for i in range(self.layers - 2):
                    x = F.relu(self.linears[i](x))
                    x = self.dropout(x)
                x = self.linears[-1](x)
                return x

            @staticmethod
            def batchify(n, b):
                ls = []
                for i in range(n):
                    if i % b == 0:
                        ls.append([i])
                    else:
                        ls[-1].append(i)

                return ls


        # simulate model variants
        def simulate(xtrain, ytrain, xval, yval, xtest, ytest, layers_config, batch, epochs, drop_prob, lr):
            print("TRAINING WITH PARAMETERS:")
            print("     layers configuration: {0}".format(layers_config))
            print("     batch size: {0}".format(batch))
            print("     droput probability: {0}".format(drop_prob))
            print("     learning rate: {0}".format(lr))
            print()

            n = len(xtrain)

            net = Net(layers_config, drop_prob)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            intervals = net.batchify(n, batch)
            tr_loss, val_loss = [], []

            best_val_loss = 1000000
            for epoch in range(epochs):

                # shuffle training set
                perm = torch.randperm(xtrain.size()[0])
                xtrain = xtrain[perm]
                ytrain = ytrain[perm]

                for i in intervals:
                    # get the inputs and labels of training set
                    inputs, labels = xtrain[i], ytrain[i]

                    # reset the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward propagation
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # evaluate the current model
                # without computing the gradients for these operations
                with torch.no_grad():
                    outputs = net(xtrain)
                    loss = criterion(outputs, ytrain)
                    tr_loss.append(loss)

                    outputs = net(xval)
                    loss = criterion(outputs, yval)
                    val_loss.append(loss)
                    if epoch == 0 or loss < best_val_loss:
                        best_val_loss = loss
                        spr_pred = (torch.exp(outputs).cpu().detach().numpy(), torch.exp(yval).cpu().detach().numpy())

                print("epoch: ", epoch+1, "train loss: ", tr_loss[-1])

            with torch.no_grad():
                outputs = net(xtest)
                test_loss = criterion(outputs, ytest)

                test_pred = (torch.exp(outputs).cpu().detach().numpy(), torch.exp(ytest).cpu().detach().numpy())

            print()
            print("Best Validation Loss:", best_val_loss / 2)
            print("Test Loss:", test_loss / 2)
            # np.savetxt("output-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy", spr_pred[0])
            # np.savetxt("yval-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy", spr_pred[1])
            np.savetxt("output_test-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy", test_pred[0])
            np.savetxt("y_test-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy", test_pred[1])
            # np.savetxt("best_validation_loss-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".npy", np.array([best_val_loss/2]))

            plt.plot(range(epochs), tr_loss)
            plt.plot(range(epochs), val_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(["Training loss", "Validation loss"])
            plt.savefig("dataset/Loss-" + str(DATASET) + "-" + str(NUM_ENTRIES))
            plt.clf()

            # add the data of this model variant
            # in order to get the best hyperparameters
            global data
            data.append([str(layers_config),
                         str(batch),
                         str(drop_prob),
                         str(lr),
                         tr_loss[-1].item(),
                         val_loss[-1].item()])


        # # initial model
        # layers_configuration = [[n_features, 6, 3, 1]]
        # batch_size_and_epochs = [(16, 20)]
        # droput_probabilities = [0]
        # learning_rate = [0.001]
        #
        # # hyperparameters settings
        # layers_configuration = [[n_features, 6, 1], [n_features, 6, 3, 1], [n_features, 6, 6, 3, 1]]
        # batch_size_and_epochs = [(1, 10), (16, 20), (200, NUM_ENTRIES)]
        # droput_probabilities = [0, 0.5]
        # learning_rate = [0.01, 0.001, 0.0001]

        # best model
        layers_configuration = [[n_features, 6, 3, 1]]
        batch_size_and_epochs = [(16, 20)]
        droput_probabilities = [0.5]
        learning_rate = [0.001]

        df = pd.DataFrame(columns=['layers_config',
                                   'batch_size',
                                   'dropout',
                                   'learning_rate',
                                   'tr_final_loss',
                                   'val_final_loss'])
        data = []

        for lc in layers_configuration:
            for bs, eps in batch_size_and_epochs:
                for dp in droput_probabilities:
                    for lr in learning_rate:
                        simulate(xtrain_indexed,
                                 ytrain,
                                 xval_indexed,
                                 yval,
                                 xtest_indexed,
                                 ytest,
                                 lc,
                                 bs,
                                 eps,
                                 dp,
                                 lr)

        for i, d in enumerate(data):
            df.loc[i] = data[i]

        df.to_csv("dataset/results-" + str(DATASET) + "-" + str(NUM_ENTRIES) + ".csv", index=False)
        print()
        print()
