# Author: Hanchen Wang (hw501@cam.ac.uk)
# Ref: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py

import sys, pdb, tensorflow as tf
sys.path.append('../')
from utils import tf_util
from train_cls_dgcnn_torchloader import NUM_CLASSES, BATCH_SIZE, NUM_POINT


class Model:
    def __init__(self, inputs, npts, labels, is_training, **kwargs):
        self.__dict__.update(kwargs)  # have self.bn_decay
        self.knn = 20
        self.is_training = is_training
        self.features = self.create_encoder(inputs)
        self.pred = self.create_decoder(self.features)
        self.loss = self.create_loss(self.pred, labels)

    @staticmethod
    def get_graph_feature(x, k):
        """Torch: get_graph_feature = TF: adj_matrix + nn_idx + edge_feature"""
        adj_matrix = tf_util.pairwise_distance(x)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        x = tf_util.get_edge_feature(x, nn_idx=nn_idx, k=k)
        return x

    def create_encoder(self, point_cloud):
        point_cloud = tf.reshape(point_cloud, (BATCH_SIZE, NUM_POINT, 3))

        ''' Previous Solution Author Provided '''
        # point_cloud_transformed = point_cloud
        # adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
        # nn_idx = tf_util.knn(adj_matrix, k=self.knn)
        # x = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=self.knn)

        x = self.get_graph_feature(point_cloud, self.knn)
        x = tf_util.conv2d(x, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, bias=False, is_training=self.is_training,
                           activation_fn=tf.nn.leaky_relu, scope='conv1', bn_decay=self.bn_decay)
        x1 = tf.reduce_max(x, axis=-2, keep_dims=True)

        x = self.get_graph_feature(x1, self.knn)
        x = tf_util.conv2d(x, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, bias=False, is_training=self.is_training,
                           activation_fn=tf.nn.leaky_relu, scope='conv2', bn_decay=self.bn_decay)
        x2 = tf.reduce_max(x, axis=-2, keep_dims=True)

        x = self.get_graph_feature(x2, self.knn)
        x = tf_util.conv2d(x, 128, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, bias=False, is_training=self.is_training,
                           activation_fn=tf.nn.leaky_relu, scope='conv3', bn_decay=self.bn_decay)
        x3 = tf.reduce_max(x, axis=-2, keep_dims=True)

        x = self.get_graph_feature(x3, self.knn)
        x = tf_util.conv2d(x, 256, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, bias=False, is_training=self.is_training,
                           activation_fn=tf.nn.leaky_relu, scope='conv4', bn_decay=self.bn_decay)
        x4 = tf.reduce_max(x, axis=-2, keep_dims=True)

        x = tf_util.conv2d(tf.concat([x1, x2, x3, x4], axis=-1), 1024, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=True, bias=False, is_training=self.is_training,
                           activation_fn=tf.nn.leaky_relu, scope='agg', bn_decay=self.bn_decay)

        x1 = tf.reduce_max(x, axis=1, keep_dims=True)
        x2 = tf.reduce_mean(x, axis=1, keep_dims=True)
        # pdb.set_trace()
        features = tf.reshape(tf.concat([x1, x2], axis=-1), [BATCH_SIZE, -1])
        return features

    def create_decoder(self, features):
        """fully connected layers for classification with dropout"""

        with tf.variable_scope('decoder_cls', reuse=tf.AUTO_REUSE):
            # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
            features = tf_util.fully_connected(features, 512, bn=True, bias=False,
                                               activation_fn=tf.nn.leaky_relu,
                                               scope='linear1', is_training=self.is_training)
            features = tf_util.dropout(features, keep_prob=0.5, scope='dp1', is_training=self.is_training)

            # self.linear2 = nn.Linear(512, 256)
            features = tf_util.fully_connected(features, 256, bn=True, bias=True,
                                               activation_fn=tf.nn.leaky_relu,
                                               scope='linear2', is_training=self.is_training)
            features = tf_util.dropout(features, keep_prob=0.5, scope='dp2', is_training=self.is_training)

            # self.linear3 = nn.Linear(256, output_channels)
            pred = tf_util.fully_connected(features, NUM_CLASSES, bn=False, bias=True,
                                           activation_fn=None,
                                           scope='linear3', is_training=self.is_training)
        return pred

    @staticmethod
    def create_loss(pred, label, smoothing=True):
        # if smoothing:
        # 	eps = 0.2
        # 	n_class = pred.size(1)
        #
        # 	one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # 	one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # 	log_prb = F.log_softmax(pred, dim=1)
        #
        # 	loss = -(one_hot * log_prb).sum(dim=1).mean()

        if smoothing:
            eps = 0.2
            # pdb.set_trace()
            one_hot = tf.one_hot(indices=label, depth=NUM_CLASSES)
            # tf.print(one_hot, output_stream=sys.stderr)  # not working
            # pdb.set_trace()
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (NUM_CLASSES - 1)
            log_prb = tf.nn.log_softmax(logits=pred, axis=1)
            # pdb.set_trace()
            cls_loss = -tf.reduce_mean(tf.reduce_sum(one_hot * log_prb, axis=1))
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
            cls_loss = tf.reduce_mean(loss)

        tf.summary.scalar('classification loss', cls_loss)

        return cls_loss


if __name__ == '__main__':

    batch_size, num_cls = 16, NUM_CLASSES
    lr_clip, base_lr, lr_decay_steps, lr_decay_rate = 1e-6, 1e-4, 50000, .7

    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    npts_pl = tf.placeholder(tf.int32, (batch_size,), 'num_points')
    labels_pl = tf.placeholder(tf.int32, (batch_size,), 'ground_truths')
    learning_rate = tf.train.exponential_decay(base_lr, global_step, lr_decay_steps, lr_decay_rate,
                                               staircase=True, name='lr')
    learning_rate = tf.maximum(learning_rate, lr_clip)

    model = Model(inputs_pl, npts_pl, labels_pl, is_training_pl)
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    print('\n\n\n==========')
    print('pred', model.pred)
    print('loss', model.loss)
    # seems like different from the what the paper has claimed:
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Init Weights
    init = tf.global_variables_initializer()
    sess.run(init, {is_training_pl: True})  # restore will cover the random initialized parameters

    for idx, var in enumerate(tf.trainable_variables()):
        print(idx, var)
