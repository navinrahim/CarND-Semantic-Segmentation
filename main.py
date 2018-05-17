import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the Saved model and the layers
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #Scaling
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001, name='new_pool3_out_scaled')
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01, name='new_pool4_out_scaled')

    # Upsample vgg_layer7_out 2x times
    layer7_up = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, 4, 2, padding='same', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),name='new_layer7_up')

    # Change depth of vgg_layer4_out to num_classes
    conv1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='same', 
                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                 kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),name='new_layer4')

    # Skip Layer
    out1 = tf.add(layer7_up, conv1,name='new_skip1')

    # Upsample out1 2x times
    dconv2 = tf.layers.conv2d_transpose(out1, num_classes, 4, 2, padding='same', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),name='new_skip1_up')

    # Change depth of vgg_layer3_out to num_classes
    conv1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same', 
                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                 kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),name='new_layer3')

    # Skip Layer
    out1 = tf.add(dconv2, conv1, name='new_skip2')

    # Upsample 8x times
    logits = tf.layers.conv2d_transpose(out1, num_classes, 16, 8, padding='same', 
                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                 kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1), name='new_logits')
    
    return logits
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    # Loss calculation
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))    

    #Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print('Training...')
    print()

    for epoch in range(epochs):
        print('*****************************')
        print('Epoch ',epoch,':')
        print('----------------')
        for X_batch , y_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                keep_prob: 0.5,
                learning_rate: 0.001
            })
        print('Loss :', loss)
    
    print('*****************************')

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    correct_label = tf.placeholder(tf.float32, (None, None, None, num_classes))
    epochs = 10
    batch_size = 8
    keep_prob = 0.5
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        logits = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(logits, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, 
                    correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 1., image_input)

        #Saving Model
        if "saved_model" in os.listdir(os.getcwd()):
            shutil.rmtree("./saved_model")

        builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
        builder.add_meta_graph_and_variables(sess, ["vgg16"])
        builder.save()

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
