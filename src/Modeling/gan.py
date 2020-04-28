import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd.variable import Variable
from torchvision import datasets
from torchvision import transforms

import arg_parser as args
from Examples.Models.GAN.utils import Logger

## global variables
BATCHSIZE = 100
NUMCLASS = 2


def images_to_vectors(images):
    # return images.view(images.size(0), 784)
    return images.view(images.size(0), 1433)


def vectors_to_images(vectors):
    # return vectors.view(vectors.size(0), 1, 28, 28)

    # here>> RuntimeError: shape '[16, 1]' is invalid for input of size 22928
    return vectors.view(vectors.size(0), 1)


class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
            # n_features = 784
        # n_features = 1433
        n_features = 16
        n_out = 1

        # self.label_embedding = torch.nn.Embedding(NUMCLASS,NUMCLASS)
        # n_features_and_class = n_features  + NUMCLASS

        self.hidden = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(n_features, n_out),
            torch.nn.Sigmoid()
        )

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(n_features, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.3)
        # )
        # self.out = nn.Sequential(
        #     torch.nn.Linear(256, n_out),
        #     torch.nn.Sigmoid()
        # )

    # def forward(self, x, labels):
    def forward(self, x):
        # d_in = torch.cat(
        #     (x, self.label_embedding(labels)), -1)
        d_in = x
        d_in = self.hidden(d_in)
        d_in = self.hidden(d_in)
        d_in = self.hidden(d_in)
        d_in = self.hidden(d_in)
        d_in = self.out(d_in)
        return d_in

        # x = self.hidden0(x)
        # x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.out(x)
        # return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        # n_features = 1433
        n_features = 16
        # n_out = 784

        # n_features_and_class = n_features + NUMCLASS
        # self.label_emb = torch.nn.Embedding(NUMCLASS,NUMCLASS )

        n_features_and_class = n_features  + NUMCLASS
        self.hidden = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(0.2)
        )

        # self.hidden0 = nn.Sequential(
        #     nn.Linear(n_features, 256),
        #     nn.LeakyReLU(0.2)
        # )
        # self.hidden1 = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2)
        # )
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2)
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(1024, n_out),
        #     nn.Tanh()
        # )

    # def forward(self, x, labels):
    def forward(self, x):
        # gen_input = torch.cat((self.label_emb(labels), x), -1)
        gen_input = x
        gen_input = self.hidden(gen_input)
        gen_input = self.hidden(gen_input)
        gen_input = self.hidden(gen_input)
        gen_input = self.hidden(gen_input)
        return gen_input

        # x = self.hidden0(x)
        # x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.out(x)
        # return x


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    # n = Variable(torch.randn(size, 100))
    # n = Variable(torch.randn(size, 1433))
    n = Variable(torch.randn(size, 16))
    return n


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         transforms.Normalize((.5,), (.5,))
         ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose,
                          download=True)



def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
               Line2D([0], [0], color="b", lw=4),
               Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
class GAN:
    def __init__(self, data, data_loader):
        self.data = data
        # self.data_loader = data_loader
        self.d1_loss_hist = []
        self.d2_loss_hist = []
        self.g_loss_hist = []
        self.num_batches = 1

    def train_generator(self, optimizer, fake_data):

        print('running train_generator..')
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = self.discriminator(fake_data)

        # Calculate error and backpropagate
        error = self.loss(prediction, ones_target(N))

        # print('in train_generator')
        error.backward()
        # plot_grad_flow(self.discriminator.named_parameters())
        # Update weights with gradients
        optimizer.step()
        # Return error
        self.g_loss_hist.append(error.tolist())
        return error

    def save_loss_to_file(self, folder='../Output/run_gan/'):
        #create directory if not alreayd exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.d1_loss_hist_pd = pd.DataFrame(self.d1_loss_hist)
        self.d2_loss_hist_pd = pd.DataFrame(self.d2_loss_hist)
        self.g_loss_hist_pd = pd.DataFrame(self.g_loss_hist)
        self.d1_loss_hist_pd.to_csv(f'{folder}d1_loss_hist.to_csv', index=False, header=False)
        self.d2_loss_hist_pd.to_csv(f'{folder}d2_loss_hist.to_csv', index=False,header=False)
        self.g_loss_hist_pd.to_csv(f'{folder}g_loss_hist.to_csv', index=False,header=False)

    def train_discriminator(self, optimizer, real_data, fake_data):
        print('running train_discriminator..')
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        # TODO what do I pass in as label of real_data?
        # gen_labels => real_label which is all 0. (minority class label)
        prediction_real = self.discriminator(real_data)
        error_real = self.loss(prediction_real, ones_target(N))

        # TODO figure out whwy retain_graph is required here when other .backward() does not required it
        error_real.backward(retain_graph=True) # error here

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, zeros_target(N))
        # print('in train discriminator error_fake')
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

        self.d1_loss_hist.append(error_real.tolist())
        self.d2_loss_hist.append(error_fake.tolist())

        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake

    def init_gan(self):
        self.discriminator = DiscriminatorNet()
        self.generator = GeneratorNet()
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      lr=0.0002)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.loss = nn.BCELoss()

    # def run_gan_once(self, batch_ind, x):
    #
    #     # edge_index = edge_index[1]
    #     # test_mask = test_mask[1]
    #     # train_mask = train_mask[1]
    #     # val_mask = val_mask[1]
    #     # x = x[1]
    #     # y = y[1]
    #
    #     # N = batch_ind.size(0)
    #     N = x.size(0)
    #
    #     # 1. Train Discriminator
    #     real_data = Variable(images_to_vectors(x))
    #     # Generate fake data and detach
    #     # (so gradients are not calculated for generator)
    #     fake_data = self.generator(noise(N)).detach()
    #     # Train D
    #     d_error, d_pred_real, d_pred_fake = \
    #         self.train_discriminator(self.d_optimizer, real_data, fake_data)
    #
    #     # 2. Train Generator
    #     # Generate fake data
    #     fake_data = self.generator(noise(N))
    #
    #     # Train G
    #     g_error = self.train_generator(self.g_optimizer, fake_data)
    #     # Log batch error

    def get_gen_labels_for_real_fake_minority_class(self, batch_size):
        return Variable(torch.LongTensor(np.random.randint(0, NUMCLASS, batch_size)))

    def run_gan(self):
        # # Load data
        # data = self.mnist_data()

        self.init_gan()

        # Num batches
        # num_batches = len(self.data_loader)
        num_batches = self.num_batches

        num_test_samples = 16
        test_noise = noise(num_test_samples)

        logger = Logger(model_name='VGAN', data_name='MNIST')
        # Total number of epochs to train
        num_epochs = 200
        for epoch in range(num_epochs):

            n_batch = 0
            edge_index = self.data.edge_index
            test_mask = self.data.test_mask
            train_mask = self.data.train_mask
            val_mask = self.data.val_mask
            x = self.data.x
            y = self.data.y

            # batch_ind = batch_ind[1]

            # edge_index = edge_index[1]
            # test_mask = test_mask[1]
            # train_mask = train_mask[1]
            # val_mask = val_mask[1]
            # x = x[1]
            # y = y[1]

            # self.run_gan_once(batch_ind, x)

            # N = batch_ind.size(0)
            N = x.size(0)
            # gen_labels = self.get_gen_labels_for_real_fake_minority_class(N)

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(x))
            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = self.generator(noise(N)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                self.train_discriminator(self.d_optimizer, real_data,
                                         fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = self.generator(noise(N))

            # Train G
            g_error = self.train_generator(self.g_optimizer, fake_data)
            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            # Display Progress every few batches

            self.save_loss_to_file()

            if (n_batch) % 100 == 0:
                # test_images = vectors_to_images(self.generator(test_noise))
                # test_images = test_images.data
                # logger.log_images(
                #     test_images, num_test_samples,
                #     epoch, n_batch, num_batches
                # )

                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake)


if __name__ == '__main__':
    # TODO here>> paste one with cora dataset instead
    dtype = torch.float
    # device = torch.device("gpu")

    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    # BATCHSIZE = 100
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCHSIZE,
                                              shuffle=True)

    gan = GAN(data, data_loader)
    gan.run_gan()
    # run_gan()
