import Utils
from ImagesAndMetadata import ImagesAndMetadata
import numpy as np
from numpy import array
import sys
import tensorflow as tf
import math
def prStats(predicted,actual):

    #predicted_1 = tf.cast(predicted > 0, predicted.dtype)
    tp_a,fp_a,fn_a = stats(predicted,actual)
    p_a,r_a,f1_a = pr(tp_a,fp_a,fn_a)
    return tp_a,fp_a,fn_a,p_a,r_a,f1_a

def stats(predicted,actual):
	tp = tf.count_nonzero(predicted * actual , dtype=tf.float32)
	fp = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
	fn = tf.count_nonzero((predicted - 1) * actual , dtype=tf.float32)
	return tp,fp,fn

def pr(tp,fp,fn):
	precision = tf.cast((tp +0.001)/ (tp + fp+0.002), dtype=tf.float32)
	recall = tf.cast((tp +0.001)/ (tp + fn+0.002), dtype=tf.float32)
	f1 = tf.cast(2 * precision * recall / (precision + recall), dtype=tf.float32)
	return precision,recall,f1

def accuracy(predicted,actual):
    return tf.reduce_sum(tf.multiply(predicted,actual))

def Run(train_meta_file, train_image_dir, test_meta_file, test_image_dir,validation_meta_file,validation_image_dir, batch_size, epoch, learning_rate, num_CNN_filters, operation, restore_validated_model, debug_logs):
    learning_rate_decay = 0.995
    classes = 14
    train_meta_data, train_labels = Utils.load_csv_file_without_images( train_meta_file, classes,debug_logs)
    print("Meta data shape ="+str(train_meta_data.shape))
    trainImagesAndMetaData = ImagesAndMetadata(train_meta_data, train_labels)
    num_images = trainImagesAndMetaData.num_images
    # placeholders for NN
    input_image_data = tf.placeholder(tf.float32, [None, 1024*1024], name="image_data")
    input_meta_data = tf.placeholder(tf.float32, [None, train_meta_data.shape[1]-1], name="meta_data")
    labels_ = tf.placeholder(tf.float32, [None, classes])

    # CNN layer
    input_layer = tf.reshape(input_image_data, [-1, 1024, 1024, 1])
    input_layer = tf.cast(input_layer, tf.float32)
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=num_CNN_filters,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    pool4_flat = tf.reshape(pool4, [-1, 64 * 64 * 64])
    Utils.WriteLog(debug_logs,"train DNN ip size=" + str(pool4_flat.shape))
    #Add the other metadata
    image_and_meta_data = tf.concat([pool4_flat, input_meta_data], 1)
    # Create the model
    featureCount = int(image_and_meta_data.shape[1]) 
    print("feature count ="+str(featureCount))
    # Define loss and optimizer

    pkeep = tf.placeholder(tf.float32,name="pkeep")
    H1 = 1000
    H2 = 100
    w1 = tf.Variable(tf.truncated_normal([featureCount, H1], stddev=0.1), name="w1")
    b1 = tf.Variable(tf.zeros([H1]), name="b1")
    w2 = tf.Variable(tf.truncated_normal([H1, H2], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.zeros([H2]), name="b2")
    w3 = tf.Variable(tf.truncated_normal([H2, classes], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.zeros([classes]), name="b3")

    y1 = tf.nn.relu(tf.matmul(image_and_meta_data, w1, name="m1") + b1, name="y1")
    y2 = tf.nn.relu(tf.matmul(y1, w2, name="m2") + b2, name="y2")
    #y2d = tf.nn.dropout(y2, pkeep)
    y = tf.nn.softmax(tf.matmul(y2, w3, name="m1") + b3, name="y")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels_)
    #cross_entropy = tf.losses.softmax_cross_entropy
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    #tp,fp,fn,p,r,f1 = prStats(y,labels_)
    acc = accuracy(y,labels_)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    iterPerEpoch = int(math.ceil(num_images/batch_size))
    print ("Iteration per Epoch  " + str(iterPerEpoch))

    test_meta_data, test_labels = Utils.load_csv_file_without_images( train_meta_file, classes,debug_logs) ## testing with traing data itself
    testImagesAndMetaData = ImagesAndMetadata( test_meta_data, test_labels)

    for i in range(epoch):
        for j in range(iterPerEpoch):
            print("At epoch "+str(i) +" iteration "+str(j)) 
            train_images_xs, train_metadata_xs, train_ys = trainImagesAndMetaData.next_batch( batch_size, str(train_image_dir))
            _,loss = sess.run([optimizer,cost], feed_dict={input_image_data: train_images_xs, input_meta_data: train_metadata_xs, labels_: train_ys,pkeep:0.75})
        print('Train Loss : ' + str(loss))

        test_images_xs, test_metadata_xs, test_ys = testImagesAndMetaData.next_batch( batch_size, str(test_image_dir))
        #test_y, test_tp,test_fp,test_fn,test_p,test_r,test_f1 = sess.run([y, tp,fp,fn,p,r,f1], feed_dict={input_image_data: test_images_xs, input_meta_data: test_metadata_xs, labels_: test_ys,pkeep:1.0})
        test_y, test_acc = sess.run([y, acc], feed_dict={input_image_data: test_images_xs, input_meta_data: test_metadata_xs, labels_: test_ys,pkeep:1.0})
        np.set_printoptions(threshold=np.nan)
        print("Actual values = "+str(test_ys))
        print("Predicted values = "+str(test_y))
        #print('Test PR  : ' + str(test_tp) + '    ' + str(test_fp) + '    ' + str(test_fn)+ '    ' + str(test_p) + '    ' + str(test_r)+ '    ' + str(test_f1))
        print('Test Accuracy  : ' + str(test_acc))
if __name__ == "__main__":
    args = sys.argv
    train_meta_file = args[1]
    train_image_dir = args[2]
    test_meta_file = args[3]
    test_image_dir = args[4]
    validation_meta_file = args[5]
    validation_image_dir = args[6]
    batch_size = int(args[7])
    epoch = int(args[8])
    learning_rate = float(args[9])
    operation = args[10] #test/train/both
    restore_validated_model = args[11] #yes/no
    debug_log_file = args[12]
    num_CNN_filters = int(args[13])
    debug_logs = open(debug_log_file, 'w')
    #np.set_printoptions(threshold='nan')
    Run(train_meta_file, train_image_dir,  test_meta_file, test_image_dir,validation_meta_file,validation_image_dir, batch_size, epoch, learning_rate, num_CNN_filters, operation, restore_validated_model, debug_logs)
    debug_logs.close()
    #print("Enter any char...")
    #name = sys.stdin.readline()
    # "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\train_sample2.csv" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\sampleImages"  "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\train_sample2.csv" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\sampleImages"  "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\train_sample2.csv" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\sampleImages" 3 5 0.02 both yes "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\log1.txt" 32
    #"C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\XRayDiseasePrediction\\data\\train_1000.csv" "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\dl2_trimages\\train_"   "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\XRayDiseasePrediction\\data\\train_last_1000.csv" "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\dl2_trimages\\train_"  "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\XRayDiseasePrediction\\data\\train__last_1000.csv" "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\dl2_trimages\\train_" 50 50 0.02 both yes "C:\\Users\\rajgup\\Documents\\XRayDiseasePrediction\\XRayDiseasePrediction\\data\\log1.txt" 32
