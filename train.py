import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from dataset import RandomMoveBoardManhattanDataset


class CNN(chainer.Chain):
    def __init__(self, w, h):
        super(CNN, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(w*h, 3)
            self.cnn1 = L.Convolution2D(3, 32, 3, pad=1)
            self.cnn2 = L.Convolution2D(None, 32, 3, pad=1)
            self.cnn3 = L.Convolution2D(None, 64, 3, pad=1)
            self.cnn4 = L.Convolution2D(None, 64, 3, pad=1)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(None, 1)
            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(64)
            self.bn5 = L.BatchNormalization(256)
    
    def __call__(self, x):
        h = self.emb(x)
        h = F.transpose(h, (0, 3, 1, 2))
        #print(h.shape)
        h = F.relu(self.bn1(self.cnn1(h)))
        h = F.relu(self.bn2(self.cnn2(h)))
        #print(h.shape)
        h = F.relu(self.bn3(self.cnn3(h)))
        h = F.relu(self.bn4(self.cnn4(h)))
        h = F.reshape(h, (h.data.shape[0], -1))
        h = F.relu(self.bn5(self.fc1(h)))
        h = self.fc2(h)
        h = F.reshape(h, (h.data.shape[0], ))

        return h


def main():
    parser = argparse.ArgumentParser(description='Chainer example: OCR')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--width', default=6, type=int)
    parser.add_argument('--height', default=6, type=int)
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = CNN(args.width, args.height)
    def loss_fun(x, t):
        print(x.shape, t.shape)
    model = L.Classifier(model, accfun=F.mean_absolute_error, lossfun=F.mean_absolute_error)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = RandomMoveBoardManhattanDataset(args.width, args.height, num=100000), RandomMoveBoardManhattanDataset(args.width, args.height, num=100)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    def savefun(path, x):
        x = x.copy()
        x.to_cpu()
        chainer.serializers.save_npz(path, x)
    trainer.extend(extensions.snapshot_object(model, 'model-%s-%s-iter_{.updater.iteration}' % (args.width, args.height), savefun=savefun), trigger=(1, 'epoch'))

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

