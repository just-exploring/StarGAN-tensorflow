import numpy as np
import os
import time
import datetime
#from PIL import Image
import tensorflow as tf
import model
tfgan = tf.contrib.gan

########################################################
# solver.py:  A simple conversion from the PyTorch
#   version by yunjey to a TensorFlow implementation
#
# Reference to the second dataset, RaFD, has been 
#   removed.
#
# Original at: https://github.com/yunjey/StarGAN
########################################################


class Solver(object):

    def __init__(self, sess, celebA_loader, config):
        # Session
        self.sess = sess

        # Data loader
        self.celebA_loader = celebA_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    #####################################################################
    def build_model(self):
        # Define a generator and a discriminator

        self.G = model.generator(images, self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = model.discriminator(images, self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        # Optimizers
        self.g_optimizer = tf.train.AdamOptimizer(self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = tf.train.AdamOptimizer(self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if tf.test.is_gpu_available(cuda_only=True):
            self.G.cuda()
            self.D.cuda()

    #####################################################################
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    #####################################################################
    def load_pretrained_model(self):
        tf.saved_model.loader.load(sess, os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model)))
        tf.saved_model.loader.load(sess, os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model)))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    #####################################################################
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    #####################################################################
    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    #####################################################################
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    #####################################################################
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    #####################################################################
    def threshold(self, x):
        x = tf.contrib.graph_editor.copy(x)
        x = (x >= 0.5).float()
        x = (x < 0.5).float()
        return x

    #####################################################################
    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = tf.metrics.mean(correct, dim=0) * 100.0
        else:
            _, predicted = tf.maximum(x, dim=1)
            correct = (predicted == y).float()
            accuracy = tf.metrics.mean(correct) * 100.0
        return accuracy

    #####################################################################
    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = tf.zeros([batch_size, dim])
        out[np.arange(batch_size), labels.long()] = 1
        return out

    #####################################################################
    def make_celeb_labels(self, real_c):
        NUM_ATTRIBUTES = 3
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = tf.eye(NUM_ATTRIBUTES) # An identity matrix for black, blonde, and brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = tf.contrib.graph_editor.copy(real_c)
            for c in fixed_c:
                if i < NUM_ATTRIBUTES:
                    c[:NUM_ATTRIBUTES] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.sess.run(fixed_c))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = tf.contrib.graph_editor.copy(real_c)
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color to brown
                        c[:3] = y[2] 
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if c[3] == 1 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if c[4] == 1 else 1
                fixed_c_list.append(self.sess.run(fixed_c))
        return fixed_c_list

    #####################################################################
    def train(self):
        """Train StarGAN within a single dataset."""
        
        # Set dataloader
        if self.dataset == 'CelebA':
            self.data_loader = self.celebA_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = tf.concat(fixed_x, 0)
        fixed_x = self.sess.run([fixed_x])
        real_c = tf.concat(real_c, 0)

        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader):
                
                # Generat fake labels randomly (target domain labels)
                rand_idx = np.random.permutation(real_label.size(0))
                fake_label = real_label[rand_idx]

                if self.dataset == 'CelebA':
                    real_c = tf.contrib.graph_editor.copy(real_label)
                    fake_c = tf.contrib.graph_editor.copy(fake_label)
                else:
                    real_c = self.one_hot(real_label, self.c_dim)
                    fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.sess.run(real_x)
                real_c = self.sess.run(real_c)           # input for the generator
                fake_c = self.sess.run(fake_c)
                real_label = self.sess.run(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.sess.run(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - tf.metrics.mean(out_src)

                if self.dataset == 'CelebA':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                    else:
                        print('Classification Acc (8 emotional expressions): ', end='')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = tf.metrics.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = tf.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = tf.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=tf.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = tf.sqrt(tf.add_n(grad ** 2, dim=1))
                d_loss_gp = tf.metrics.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c)
                    rec_x = self.G(fake_x, real_c)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - tf.metrics.mean(out_src)
                    g_loss_rec = tf.metrics.mean(tf.abs(real_x - rec_x))

                    if self.dataset == 'CelebA':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.G(fixed_x, fixed_c))
                    fake_images = tf.concat(fake_image_list, dim=3)
                    scipy.misc.imsave(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    self.checkpoint_save
                    self.sample_save
                    tf.saved_model.loader.load(sess, os.path.join(
                        self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    tf.saved_model.loader.load(sess, 
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    #####################################################################
    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        tf.saved_model.loader.load(self.sess, G_path)
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.celebA_loader

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.sess.run(real_x)

            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(org_c)
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(tf.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.sess.run(target_c))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = tf.concat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))
