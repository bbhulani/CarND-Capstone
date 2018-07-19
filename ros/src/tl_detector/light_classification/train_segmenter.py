import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph = tf.get_default_graph()

    image_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    initialization_stddev = 1e-3
    weight_penalty = 1e-3

    # 1x1 convolutions on vgg layer7,4,3 to match training class dimensionality
    # (e.g. keep other dimensions the same but reduce dimensionality of the output
    # filters down to num_classes filters)
    vgg_layer7_out_match_num_classes = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))

    vgg_layer4_out_match_num_classes = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))

    vgg_layer3_out_match_num_classes = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))

    # deconvolve - learn weights which upsample 2x to match vgg_layer4_out_match_num_classes dimensions
    # then implement skip layer to include features at scale of vgg_layer4
    layer7_to_layer4_upsample = tf.layers.conv2d_transpose(
        vgg_layer7_out_match_num_classes,
        num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))

    skip_layer_4 = tf.add(layer7_to_layer4_upsample, vgg_layer4_out_match_num_classes)

    # deconvolve - learn weights which upsample 2x to match vgg_layer3 dimensions
    # then implement skip layer to include features at scale of vgg_layer3
    layer4_to_layer3_upsample = tf.layers.conv2d_transpose(
        skip_layer_4,
        num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))

    skip_layer_3 = tf.add(layer4_to_layer3_upsample, vgg_layer3_out_match_num_classes)

    # deconvolve -- learn weights which upsample 4x to match original image dimensions
    layer3_to_inputimage_upsample = tf.layers.conv2d_transpose(
        skip_layer_3,
        num_classes,
        kernel_size=16,
        strides=(8, 8),
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty))
    tf.Print(layer3_to_inputimage_upsample, [tf.shape(layer3_to_inputimage_upsample)])
    return layer3_to_inputimage_upsample


def optimize(nn_last_layer,
             correct_label,
             learning_rate,
             num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    # labels = tf.reshape(correct_label, (-1, num_classes))

    logits = tf.identity(nn_last_layer, name="logits")
    labels = correct_label

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Apply L2 Regularization to penalize large weights (on operations in the graph with a kernel_regularizer)
    loss = tf.add(cross_entropy_loss, tf.losses.get_regularization_loss())
    # loss = cross_entropy_loss

    # get_regularization_loss magic explanation:
    # Suppose this node in graph:
    # => vgg_layer7_out_match_num_classes = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(decay_rate))
    # Then get_regularization_loss() evaluates to:
    # => tf.nn.l2_loss(weights_for_conv2d_kernel) * decay_rate

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess,
             epochs,
             batch_size,
             get_batches_fn,
             gen_validation_batches_fn,
             train_op,
             cross_entropy_loss,
             input_image,
             correct_label,
             keep_prob,
             learning_rate,
             checkpoint_function=None,
             training_steps=None,
             validation_steps=None
             ):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
    :param gen_validation_batches_fn: Function to get batches of validation data.
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param checkpoint_function: Optional function to serialize a checkpoint of the model

    """
    KEEP_PROBABILITY = 0.5
    LEARNING_RATE = 0.0001

    num_training_samples = get_batches_fn.num_samples
    num_validation_samples = gen_validation_batches_fn.num_samples

    if training_steps is None:
        training_steps = num_training_samples // batch_size
    if validation_steps is None:
        validation_steps = num_validation_samples // batch_size

    print("Starting training: {0} training samples {1} validation_samples -- {2} training steps {3} validation steps...".format(
        num_training_samples, num_validation_samples, training_steps, validation_steps
    ))
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        training_loss = 0

        for step, (images, labels) in enumerate(get_batches_fn(batch_size)):
            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: KEEP_PROBABILITY,
                learning_rate: LEARNING_RATE
            }
            sess.run(train_op, feed_dict=feed_dict)
            loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
            print("Training Batch [{0} of {1}] Loss={2:.6f}".format(step, training_steps, loss))
            training_loss += loss

            if step >= training_steps:
                break

        validation_loss = 0

        for step, (images, labels) in enumerate(gen_validation_batches_fn(batch_size)):
            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: KEEP_PROBABILITY,
                learning_rate: LEARNING_RATE
            }
            loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
            print("Validation Batch [{0} of {1}] Loss={2:.6f}".format(step, validation_steps, loss))
            validation_loss += loss

            if step >= validation_steps:
                break

        print("EPOCH {} ...".format(epoch+1))
        print("Training Loss For Epoch = {:.3f}".format(training_loss / training_steps))
        print("Validation Loss For Epoch = {:.3f}".format(validation_loss / validation_steps))

        if checkpoint_function:
            checkpoint_function(sess, epoch)

    print("Training finished.")


def train(dataset_parameters,
          batch_size,
          num_epochs,
          num_training_steps,
          num_validation_steps
          ):

    with tf.Session() as sess:
        # path to vgg model
        vgg_path = os.path.join(dataset_parameters.data_dir, "vgg")

        # Create functions for getting training/test batches
        gen_training_batches_fn, gen_validation_batches_fn = helper.gen_batch_function(dataset_parameters)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
            sess, vgg_path)
        result_output = layers(layer3_out, layer4_out, layer7_out, dataset_parameters.num_classes)

        # Not this -- the labels are already one-hot encoded ...
        # groundtruth_placeholder = tf.placeholder(
        #     tf.int32, shape=(None, *image_shape), name="groundtruth")
        # onehot_labels = tf.one_hot(groundtruth_placeholder, depth=num_classes)

        groundtruth_dimensions = (None,) + dataset_parameters.image_shape + (dataset_parameters.num_classes,)

        groundtruth_placeholder = tf.placeholder(tf.bool, shape=groundtruth_dimensions)
        # onehot_labels = tf.one_hot(groundtruth_placeholder, depth=num_classes)

        learning_rate_placeholder = tf.placeholder(
            tf.float32, name="learning_rate"
        )

        _, train_op, cross_entropy_loss = optimize(
            result_output, groundtruth_placeholder, learning_rate_placeholder, dataset_parameters.num_classes)

        # Train NN using the train_nn function
        EPOCHS = num_epochs
        BATCH_SIZE = batch_size
        saver = tf.train.Saver(max_to_keep=2)

        def checkpoint_function(sess, epoch):
            model_path = dataset_parameters.model_savepath()
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            saver.save(sess, model_path, global_step=epoch)
            print("Model checkpoint saved")

        train_nn(sess,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 get_batches_fn=gen_training_batches_fn,
                 gen_validation_batches_fn=gen_validation_batches_fn,
                 train_op=train_op,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=input_image,
                 correct_label=groundtruth_placeholder,
                 keep_prob=keep_prob,
                 learning_rate=learning_rate_placeholder,
                 checkpoint_function=checkpoint_function,
                 training_steps=num_training_steps,
                 validation_steps=num_validation_steps
                 )


def restore_model(sess, model_dir):
    model_meta = ".".join([tf.train.latest_checkpoint(model_dir), "meta"])
    model_data = tf.train.latest_checkpoint(model_dir)

    print("model_meta {0} model_data {1}".format(model_meta, model_data))

    saver = tf.train.import_meta_graph(model_meta)
    saver.restore(sess, model_data)

    graph = tf.get_default_graph()

    logits = graph.get_tensor_by_name("logits:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    input_image = graph.get_tensor_by_name("image_input:0")

    return logits, keep_prob, input_image


def test(dataset_parameters):
    tf.reset_default_graph()

    model_dir = dataset_parameters.model_dir()

    with tf.Session() as sess:
        logits, keep_prob, input_image = restore_model(sess, model_dir)
        print("Model restored.")

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(
            dataset_parameters,
            sess,
            logits,
            keep_prob,
            input_image)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", help="Minibatch size", default=8, type=int)
    parser.add_argument("--num-epochs", help="Number of epochs to train", default=8, type=int)
    parser.add_argument("--num-training-steps", help="Number of steps in training epoch (defaults to num_training_samples//batch_size)", default=None, type=int)
    parser.add_argument("--num-validation-steps", help="Number of steps in validation epoch (defaults to num_validation_samples//batch_size)", default=None, type=int)

    parser.add_argument("--data-dir", help="Path to directory containing labeled training images", default='../../../../../vmshared')
    parser.add_argument("--mode", help="Whether to train or categorize images", default="train", choices=["train", "test"])
    parser.add_argument("--dataset", help="Choose dataset to train or categorize", default="traffic-lights", choices=["traffic-lights", "roads"])
    # parser.add_argument("--color-mode", help="Feed grayscale or multi-channel image into classifer model?", default="grayscale")
    # parser.add_argument("--save-generated-images", help="Save the generated images for human verification", default=False, action='store_true')
    # parser.add_argument("--use-bosch", help="Train from bosch images", default=False, action="store_true")
    # parser.add_argument("--model", help="Select model to use", choices=["simple", "squeezenet"], default="simple")

    args = parser.parse_args()

    print(args)

    if args.dataset == "roads":
        dataset = helper.RoadParameters(args.data_dir)
    else:
        dataset = helper.TrafficLightParameters(args.data_dir)

    if args.mode == "train":
        train(dataset,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              num_training_steps=args.num_training_steps,
              num_validation_steps=args.num_validation_steps
              )
    elif args.mode == "test":
        test(dataset)


if __name__ == '__main__':
    main()
