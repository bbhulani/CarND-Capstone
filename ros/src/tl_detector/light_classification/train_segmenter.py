import os.path
import tensorflow as tf
import dataset as helper
import warnings
from distutils.version import LooseVersion
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))


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
    weight_penalty = 1e-4

    # 1x1 convolutions on vgg layer7,4,3 to match training class dimensionality
    # (e.g. keep other dimensions the same but reduce dimensionality of the output
    # filters down to num_classes filters)
    vgg_layer7_out_match_num_classes = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=initialization_stddev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_penalty)
    )

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

    logits = tf.identity(layer3_to_inputimage_upsample, name="logits")

    # turn logits into probabilities
    predict_label_probabilities = tf.nn.softmax(logits, name="predict_label_probabilities")

    # turn probabilities into class predictions
    predict_labels = tf.argmax(predict_label_probabilities, axis=-1)
    # predict_labels = tf.Print(predict_labels, [predict_labels], "predict_labels: ")

    # count total number of predictions in batch
    sample_count = tf.size(predict_labels)
    sample_count = tf.cast(sample_count, tf.float32)
    # sample_count = tf.Print(sample_count, [sample_count], message="Sample count: ")

    # count number of predictions for each class in batch
    predict_class_one_hot = tf.one_hot(predict_labels, num_classes, axis=-1)
    label_frequency = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(predict_class_one_hot, axis=0), axis=0), axis=0)
    label_frequency = tf.cast(label_frequency, tf.float32)
    # label_frequency = tf.Print(label_frequency, [label_frequency], message="label_frequency: ")

    # print("predict_labels.shape: ", predict_labels.shape)
    # print("predict_class_one_hot.shape: ", predict_class_one_hot.shape)
    # print("label_frequency.shape: ", label_frequency.shape)
    # print("sample_count.shape", sample_count.shape)

    # distribution by class of predictions
    predict_label_distribution = tf.divide(label_frequency, sample_count, name="predict_label_distribution")

    return logits, predict_label_probabilities, predict_label_distribution


def optimize(logits,
             predict_layer_probabilities,
             labels,
             learning_rate,
             num_classes,
             loss_function,  # "cross_entropy" or "weighted_cross_entropy" or "iou_estimate" or "weighted_iou_estimate"
             regularization_loss=10e-4  # None or constant value by which to scale regularization loss
             ):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss, metric_initializer, calc_mean_iou, update_confusion_matrix_op)
    """

    report_class_counts = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(labels, axis=1), axis=1), axis=0)

    print("report_class_counts shape", report_class_counts.get_shape())

    # a. optimize weighted cross entropy loss ...
    if loss_function == "cross_entropy":
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
    elif loss_function == "weighted_cross_entropy":
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # try to deal with the extreme class imbalance in the source dataset (vast majority of pixels in image have label 'unknown')
        # re-balance the one-hot representation of the correct labels for the image
        # to make the optimizer prefer correct red light pixel prediction more than other pixel predictions
        # class_weights = tf.constant([[.10, .30, .30, .30]], dtype=tf.float32)

        sample_count = tf.reduce_sum(labels)
        class_frequency = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(labels, axis=1), axis=1), axis=0)
        class_probability = tf.divide(class_frequency, sample_count)
        class_weights = tf.subtract(1.0, class_probability)

        # debug node to print out class weights and frequency
        # class_weights = tf.Print(class_weights, [class_weights, class_frequency], "\nclass weights and class frequency: ", summarize=1000)

        weighted_labels = tf.multiply(class_weights, labels)
        balanced_weights = tf.reduce_sum(weighted_labels, axis=3)

        print("cross_entropy shape", cross_entropy.get_shape())

        weighted_cross_entropy = tf.multiply(balanced_weights, cross_entropy)

        print("weighted_cross_entropy shape", weighted_cross_entropy.get_shape())
        loss = tf.reduce_mean(weighted_cross_entropy)
    elif loss_function == "iou_estimate":

        # per http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf equation (2) -- approximate intersection count as sum(logits * labels)
        intersection = tf.reduce_sum(tf.multiply(predict_layer_probabilities, labels))

        # per http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf equation (4) -- approximate union count as sum(logits + labels - logits*labels )
        union = tf.reduce_sum(tf.subtract(tf.add(predict_layer_probabilities, labels), tf.multiply(predict_layer_probabilities, labels)))

        # estimate iou_loss: per http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf equation (5) -- approximate loss as 1 - (intersection/union)
        loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(intersection, union))
    elif loss_function == "weighted_iou_estimate":
        sample_count = tf.reduce_sum(labels)
        class_frequency = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(labels, axis=1), axis=1), axis=0)
        class_probability = tf.divide(class_frequency, sample_count)
        class_weights = tf.subtract(1.0, class_probability)

        # debug node to print out class weights and frequency
        # class_weights = tf.Print(class_weights, [class_weights, class_frequency], "\nclass weights and class frequency: ", summarize=1000)

        weighted_probability = predict_layer_probabilities * class_weights
        weighted_labels = labels * class_weights

        # weighted version of intersection that reduces contribution from UNKNOWN pixels labels
        weighted_intersection = tf.reduce_sum(tf.multiply(weighted_probability, weighted_labels))

        # weighted version of union that reduces contribution from 'UNKNOWN' pixels
        weighted_union = tf.reduce_sum(tf.subtract(tf.add(weighted_probability, weighted_labels), tf.multiply(weighted_probability, weighted_labels)))

        # estimate iou_loss with class weighting
        # per http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf equation (5) -- approximate loss as 1 - (intersection/union)
        loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(weighted_intersection, weighted_union))
    else:
        raise Exception("Error: Unsupported loss function")

    if regularization_loss:
        loss = tf.add(loss, tf.multiply(regularization_loss, tf.losses.get_regularization_loss()))

    print("Labels shape", labels.get_shape())
    print("Logits shape", logits.get_shape())

    true_labels = tf.argmax(labels, axis=3)
    predicted_labels = tf.argmax(predict_layer_probabilities, axis=3)

    print("tf.argmax(Labels) shape", true_labels.get_shape())
    print("tf.argmax(Logits) shape", predicted_labels.get_shape())

    # Use AdamOptimizer to minimize the loss function
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    # Calculate mean_iou for a batch
    calc_mean_iou, update_confusion_matrix_op = tf.metrics.mean_iou(
        true_labels,
        predicted_labels,
        4,
        weights=None,  # tf.constant([.33, .33, .33, 0]),
        metrics_collections=None,
        updates_collections=None,
        name="metrics"
    )
    mean_iou_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

    # Define initializer to initialize/reset mean iou metrics -- this allows tracking stats a batch at a time
    metric_initializer = tf.variables_initializer(var_list=mean_iou_running_vars)

    return train_op, loss, metric_initializer, calc_mean_iou, update_confusion_matrix_op, report_class_counts


def train_nn(sess,
             epochs,
             batch_size,
             get_batches_fn,
             gen_validation_batches_fn,
             train_op,
             compute_loss,
             input_image,
             correct_label,
             keep_prob,
             learning_rate,
             logits,
             metric_initializer,
             calc_mean_iou,
             update_confusion_matrix_op,
             report_class_counts,
             checkpoint_function=None,
             training_steps=None,
             validation_steps=None,
             restore_from_checkpoint_dir=None,
             generate_test_images=None
             ):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
    :param gen_validation_batches_fn: Function to get batches of validation data.
    :param train_op: TF Operation to train the neural network
    :param compute_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param metric_initializer,
    :param calc_mean_iou,
    :param update_confusion_matrix_op,
    :param report_class_counts,
    :param checkpoint_function: Optional function to serialize a checkpoint of the model
    :param training_steps=None,
    :param validation_steps=None,
    :param restore_from_checkpoint_dir=None: Optiona, when presention restore latest checkpoint from this directory
    """
    KEEP_PROBABILITY = 0.5
    LEARNING_RATE = 10e-5

    num_training_samples = get_batches_fn.num_samples
    num_validation_samples = gen_validation_batches_fn.num_samples

    if training_steps is None:
        training_steps = num_training_samples // batch_size
    if validation_steps is None:
        validation_steps = num_validation_samples // batch_size

    print("Starting training: {0} training samples {1} validation_samples -- {2} training steps {3} validation steps...".format(
        num_training_samples, num_validation_samples, training_steps, validation_steps
    ))

    if restore_from_checkpoint_dir is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver()
        model_data = tf.train.latest_checkpoint(restore_from_checkpoint_dir)
        print("Restoring weights from checkpoint {0}".format(model_data))
        saver.restore(sess, model_data)

    for epoch in range(epochs):
        training_loss = 0
        training_mean_iou = 0

        for step, (images, labels) in enumerate(get_batches_fn(batch_size)):
            # Reset the variables in which batch metrics are stored
            sess.run(metric_initializer)

            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: KEEP_PROBABILITY,
                learning_rate: LEARNING_RATE
            }
            sess.run(train_op, feed_dict=feed_dict)
            loss, class_counts, _ = sess.run([compute_loss, report_class_counts, update_confusion_matrix_op], feed_dict=feed_dict)

            mean_iou = sess.run(calc_mean_iou)

            print("Training Batch [{0} of {1}] Loss={2:.9f} Mean_IOU: {3} Class_counts: [{4}]".format(step, training_steps, loss, mean_iou, class_counts))
            training_loss += loss
            training_mean_iou += mean_iou

            if step >= training_steps:
                break

        if generate_test_images:
            print("Generating validation images for epoch")
            generate_test_images(epoch, sess, logits, keep_prob, input_image)

        validation_loss = 0
        validation_mean_iou = 0

        for step, (images, labels) in enumerate(gen_validation_batches_fn(batch_size)):
            # Reset the variables in which batch metrics are stored
            sess.run(metric_initializer)

            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: 1.0,
                learning_rate: LEARNING_RATE,
            }

            # compute loss and update confusion matrix
            loss, class_counts, _ = sess.run([compute_loss, report_class_counts, update_confusion_matrix_op], feed_dict=feed_dict)
            validation_loss += loss

            mean_iou = sess.run(calc_mean_iou)
            validation_mean_iou += mean_iou

            print("Validation Batch [{0} of {1}] Loss={2:.9f} Mean_IOU: {3} Class_counts: [{4}]".format(step, validation_steps, loss, mean_iou, class_counts))

            if step >= validation_steps:
                break

        print("EPOCH {} ...".format(epoch+1))
        print("Training Summary For Epoch Loss = {0:.6f} Mean_IOU {1:3f} ".format(training_loss / training_steps, training_mean_iou / training_steps))
        print("Validation Summary For Epoch Loss = {0:.6f} Mean_IOU {1:3f}".format(validation_loss / validation_steps, validation_mean_iou / validation_steps))

        if checkpoint_function:
            checkpoint_function(sess, epoch)

    print("Training finished.")


def train(dataset_parameters,
          batch_size,
          num_epochs,
          num_training_steps,
          num_validation_steps,
          restore_from_checkpoint_dir,
          create_frozen=True
          ):

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn(
            'No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

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
        logits, predict_label_probabilities, predict_label_distribution = layers(layer3_out, layer4_out, layer7_out, dataset_parameters.num_classes)

        # Not this -- the labels are already one-hot encoded ...
        # groundtruth_placeholder = tf.placeholder(
        #     tf.int32, shape=(None, *image_shape), name="groundtruth")
        # onehot_labels = tf.one_hot(groundtruth_placeholder, depth=num_classes)

        # groundtruth_dimensions = (None,) + dataset_parameters.image_shape + (dataset_parameters.num_classes,)
        groundtruth_dimensions = (None, None, None, dataset_parameters.num_classes)
        groundtruth_placeholder = tf.placeholder(tf.float32, shape=groundtruth_dimensions)

        learning_rate_placeholder = tf.placeholder(
            tf.float32, name="learning_rate"
        )

        train_op, compute_loss, metric_initializer, calc_mean_iou, update_confusion_matrix_op, report_class_counts = optimize(
            logits,
            predict_label_probabilities,
            groundtruth_placeholder,
            learning_rate_placeholder,
            dataset_parameters.num_classes,
            loss_function="weighted_iou_estimate"
        )

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

            if create_frozen:
                # print(sess.graph.get_operations())

                constant_graph = tf.graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph_def,
                    ["logits", "predict_label_probabilities", "predict_label_distribution"]
                )

                tf.train.write_graph(constant_graph,
                                     model_dir,
                                     'saved_model-{0}.pb'.format(epoch),
                                     as_text=False)

            print("Model checkpoint saved")

        validation_preview_images = dataset_parameters.validation_preview_images()

        def generate_test_images(epoch, sess, logits, keep_prob, input_image):

            print("Generating validation images ", validation_preview_images)
            # Save inference data using helper.save_inference_samples
            helper.save_inference_samples(
                validation_preview_images,
                dataset_parameters,
                sess,
                logits,
                keep_prob,
                input_image,
                predict_label_probabilities,
                predict_label_distribution,
                directory_name=epoch
            )

        train_nn(sess,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 get_batches_fn=gen_training_batches_fn,
                 gen_validation_batches_fn=gen_validation_batches_fn,
                 train_op=train_op,
                 compute_loss=compute_loss,
                 input_image=input_image,
                 correct_label=groundtruth_placeholder,
                 keep_prob=keep_prob,
                 learning_rate=learning_rate_placeholder,
                 logits=logits,
                 metric_initializer=metric_initializer,
                 calc_mean_iou=calc_mean_iou,
                 update_confusion_matrix_op=update_confusion_matrix_op,
                 report_class_counts=report_class_counts,
                 checkpoint_function=checkpoint_function,
                 training_steps=num_training_steps,
                 validation_steps=num_validation_steps,
                 restore_from_checkpoint_dir=restore_from_checkpoint_dir,
                 generate_test_images=generate_test_images
                 )


def restore_model(sess,
                  model_dir,
                  restore_frozen=False):
    """
    Restore the model from serialized form -- if restore_frozen is given, it should
    be the string name of the frozen.pb file
    """

    graph = tf.get_default_graph()

    if restore_frozen:
        print("Restoring frozen model {0}".format(restore_frozen))
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(restore_frozen, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph

        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")

    else:
        model_meta = ".".join([tf.train.latest_checkpoint(model_dir), "meta"])
        model_data = tf.train.latest_checkpoint(model_dir)

        print("Restoring non-frozen model graph: model_meta {0} model_data {1}".format(model_meta, model_data))
        saver = tf.train.import_meta_graph(model_meta)
        saver.restore(sess, model_data)

    # retrieve the needed tensor inputs and outputs from the graph
    # print(sess.graph.get_operations())

    logits = graph.get_tensor_by_name("logits:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    input_image = graph.get_tensor_by_name("image_input:0")
    predict_label_probabilities = graph.get_tensor_by_name("predict_label_probabilities:0")
    predict_label_distribution = graph.get_tensor_by_name("predict_label_distribution:0")

    # predict_labels = tf.argmax(predict_label_probabilities, axis=-1)
    # # predict_labels = tf.Print(predict_labels, [predict_labels], "predict_labels: ")

    # sample_count = tf.size(predict_labels)
    # sample_count = tf.cast(sample_count, tf.float32)
    # # sample_count = tf.Print(sample_count, [sample_count], message="Sample count: ")

    # predict_class_one_hot = tf.one_hot(predict_labels, num_classes, axis=-1)
    # label_frequency = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(predict_class_one_hot, axis=0), axis=0), axis=0)
    # label_frequency = tf.cast(label_frequency, tf.float32)
    # # label_frequency = tf.Print(label_frequency, [label_frequency], message="label_frequency: ")

    # # print("predict_labels.shape: ", predict_labels.shape)
    # # print("predict_class_one_hot.shape: ", predict_class_one_hot.shape)
    # # print("label_frequency.shape: ", label_frequency.shape)
    # # print("sample_count.shape", sample_count.shape)

    # predict_label_distribution = tf.divide(label_frequency, sample_count, name="predict_label_distribution")

    return logits, keep_prob, input_image, predict_label_probabilities, predict_label_distribution


def test(dataset_parameters, test_image_dir=None, use_frozen_model=None):
    tf.reset_default_graph()

    model_dir, frozen_model_name = dataset_parameters.model_savedir(), None

    with tf.Session() as sess:
        logits, keep_prob, input_image, predict_label_probabilities, predict_label_distribution = restore_model(sess, model_dir, use_frozen_model)
        print("Model restored.")

        if test_image_dir is None:
            test_images, dir_name = dataset_parameters.test_images(), "dataset_test_images"
        else:
            test_images, dir_name = helper.recursive_glob(test_image_dir, "*.png", "*.jpg"), os.path.basename(os.path.normpath(test_image_dir))

        print(dir_name)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(
            test_images,
            dataset_parameters,
            sess,
            logits,
            keep_prob,
            input_image,
            predict_label_probabilities,
            predict_label_distribution,
            directory_name=dir_name
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", help="Path to directory containing labeled training images", default='../../../../../vmshared')
    parser.add_argument("--batch-size", help="Minibatch size", default=8, type=int)
    parser.add_argument("--num-epochs", help="Number of epochs to train", default=8, type=int)
    parser.add_argument("--num-training-steps", help="Number of steps in training epoch (defaults to num_training_samples//batch_size)", default=None, type=int)
    parser.add_argument("--num-validation-steps", help="Number of steps in validation epoch (defaults to num_validation_samples//batch_size)", default=None, type=int)
    parser.add_argument("--mode", help="Whether to train or categorize images", default="train", choices=["train", "test"])
    parser.add_argument("--dataset", help="Choose dataset to train or categorize", default="traffic-lights", choices=["traffic-lights", "roads"])
    parser.add_argument("--restore-from-checkpoint", help="Restore model from latest checkpoint", action='store_true')
    parser.add_argument("--test-images-from-directory", help="Test images from directory (recursive)", default=None)
    parser.add_argument("--use-frozen-model", help="Path to frozen model to test with", required=False)

    args = parser.parse_args()

    print(args)

    if args.dataset == "roads":
        dataset = helper.RoadParameters(args.data_dir)
    else:
        dataset = helper.TrafficLightParameters(args.data_dir)

    if args.restore_from_checkpoint:
        restore_from_checkpoint_dir = dataset.model_savedir()
    else:
        restore_from_checkpoint_dir = None

    if args.mode == "train":
        train(dataset,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              num_training_steps=args.num_training_steps,
              num_validation_steps=args.num_validation_steps,
              restore_from_checkpoint_dir=restore_from_checkpoint_dir
              )
    elif args.mode == "test":
        if args.use_frozen_model:
            test(dataset, args.test_images_from_directory, args.use_frozen_model)
        else:
            test(dataset, args.test_images_from_directory)


if __name__ == '__main__':
    main()
