import os
import sys
import time
import random
import shutil
import numpy as np
from PIL import Image
from scipy.misc import imsave
from tqdm import tqdm_notebook

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.layers import *
from src.model import *

class CycleGAN():
    def __init__(self,img_width = 256, img_height = 256, img_layer = 3):
        self.img_width = img_width
        self.img_height = img_height
        self.img_layer = img_layer

    def input_setup(self, src_folder):
        """
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        """

        # filenames_A = tf.train.match_filenames_once("./input/horse2zebra/trainA/*.jpg")
        folder_A = 'trainA'
        folder_B = 'trainB'
        path_A = os.path.join(src_folder, folder_A)
        path_B = os.path.join(src_folder, folder_B)
        print (' ---> TrainA :', len(os.listdir(path_A)), 'images ---- TrainB',len(os.listdir(path_A)), 'images')
        if os.listdir(path_A) and os.listdir(path_B):
            
            pattern_A = os.path.join(path_A,"*.jpg")
            print (' ---> pattern_A : ',pattern_A)
            filenames_A = tf.train.match_filenames_once(pattern_A)
            self.queue_length_A = tf.size(filenames_A)

            pattern_B = os.path.join(path_B,"*.jpg")
            print (' ---> pattern_B : ',pattern_B)
            filenames_B = tf.train.match_filenames_once(pattern_B)    
            self.queue_length_B = tf.size(filenames_B)
            
            filename_queue_A = tf.train.string_input_producer(filenames_A)
            filename_queue_B = tf.train.string_input_producer(filenames_B)

            image_reader = tf.WholeFileReader()
            _, image_file_A = image_reader.read(filename_queue_A)
            _, image_file_B = image_reader.read(filename_queue_B)

            self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[self.img_height,self.img_width]),127.5),1)
            self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[self.img_height,self.img_width]),127.5),1)

    def input_read(self, sess):
        """
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        """

        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord=coord)
        
        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((self.pool_size,1,self.img_height, self.img_width, self.img_layer))
        self.fake_images_B = np.zeros((self.pool_size,1,self.img_height, self.img_width, self.img_layer))


        self.A_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))
        self.B_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))

        with tqdm_notebook(total = len(range(self.max_images)), desc = 'ImageA', leave = False) as pbar:
            for i in range(self.max_images): 
                pbar.update(1)
                image_tensor = sess.run(self.image_A)
                # if i == 1:
                    # print (' ---> image_tensor', image_tensor, ' Type:', type(image_tensor), ' Shape:', image_tensor.shape, ' Temp:', img_size*batch_size*img_layer)
                #if(image_tensor.size() == img_size*batch_size*img_layer):
                self.A_input[i] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.img_layer))

        with tqdm_notebook(total = len(range(self.max_images)), desc = 'ImageB', leave = False) as pbar:
            for i in range(self.max_images):
                pbar.update(1)
                image_tensor = sess.run(self.image_B)
                #if(image_tensor.size() == img_size*batch_size*img_layer):
                self.B_input[i] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.img_layer))


        coord.request_stop()
        coord.join(threads)


    def model_setup(self):

        ''' This function sets up the model to train

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        # with tf.variable_scope('Inputs') as scope:
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        
        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        # for var in self.model_vars: 
        #     print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch, output_training_images_dir):

        if not os.path.exists(output_training_images_dir):
            os.makedirs(output_training_images_dir)

        for i in range(0,10):
            op_training_folder = os.path.join(output_training_images_dir,str(i))
            if not os.path.exists(op_training_folder):
                os.mkdir(op_training_folder)
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B]
                ,feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]}
            )
            imsave(op_training_folder + "/fakeB_"  + str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave(op_training_folder + "/fakeA_"  + str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave(op_training_folder + "/cycA_"   + str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            imsave(op_training_folder + "/cycB_"   + str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            imsave(op_training_folder + "/inputA_" + str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            imsave(op_training_folder + "/inputB_" + str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < self.pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,self.pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


    def train(self, src_folder, batch_size = 1, max_images = 100
                    , pool_size = 50
                    , output_path = "./output", model_summary_dir = "./output/model_summary", model_restore = False, model_restore_dir = "./output/model_checkpoints/"
                    , output_training_images = True, output_training_images_dir = './output/training_images'):
        """ 
        Training Function
        """
        self.batch_size = batch_size
        self.max_images = max_images
        self.pool_size = pool_size

        # Load Dataset from the dataset folder
        print ('1. Loading input data ...')
        self.input_setup(src_folder)  
        print ('1. Loaded input data')

        #Build the network
        print ('2. Building Model ...')
        self.model_setup()
        print ('2. Built Model ...')

        #Loss function calculations
        print ('3. Setting up loss functions ... ')
        self.loss_calc()
        print ('3. Setup loss functons')
      
        # Initializing the global variables
        print ('4. Initializing Global Variables ... ')
        init_global = tf.global_variables_initializer()
        init_local  = tf.local_variables_initializer() 
        saver = tf.train.Saver()     
        print ('4. Initialized Global Variables')

        with tf.Session() as sess:
            sess.run(init_global)
            sess.run(init_local)

            #Read input to nd array
            print ('5. Processing input data ...')
            self.input_read(sess)
            print ('6. Processed input data')

            # Model Summary
            if not os.path.exists(model_summary_dir):
                os.mkdir(model_summary_dir)
            writer = tf.summary.FileWriter(model_summary_dir)
            writer.add_graph(sess.graph)

            # Model Checkpointing
            if not os.path.exists(model_restore_dir):
                os.makedirs(model_restore_dir)
                
            if model_restore:
                chkpt_fname = tf.train.latest_checkpoint(model_restore_dir)
                saver.restore(sess, chkpt_fname)

            # Training Loop
            print ('\nTraining Begins. Epochs:', self.global_step)
            t0 = time.time()
            for epoch in range(sess.run(self.global_step),100):                
                print ("Epoch ---------------------------> ", epoch, ' (Time Elapsed:', round(time.time() - t0), 's)')
                saver.save(sess,os.path.join(model_restore_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                if(output_training_images):
                    self.save_training_images(sess, epoch, output_training_images_dir)

                t1 = time.time()
                for ptr in range(0,max_images):
                    if ptr % 10 == 0:
                        print("Iteration --------> ",ptr, ' (Time Elapsed:', round(time.time() - t0), 's)  (Batch Avg Time:', round(time.time() - t1)/10, 's)')
                        t1 = time.time()

                    # Optimizing the G_A network
                    _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})                    
                    writer.add_summary(summary_str, epoch*max_images + ptr)                    
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                    writer.add_summary(summary_str, epoch*max_images + ptr)        
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    self.num_fake_inputs+=1

                sess.run(tf.assign(self.global_step, epoch + 1))

            

    def test(self):
        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputA_"+str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputB_"+str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))