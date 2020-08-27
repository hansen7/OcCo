#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, pdb, time, argparse, datetime, importlib, numpy as np, tensorflow as tf
from utils import lmdb_dataflow, add_train_summary, plot_pcd_three_views
from termcolor import colored


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--lmdb_train', default='data/modelnet/train.lmdb')
parser.add_argument('--lmdb_valid', default='data/modelnet/test.lmdb')
parser.add_argument('--log_dir', type=str, default='')
parser.add_argument('--model_type', default='pcn_cd')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--restore_path', default='log/pcn_cd')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_gt_points', type=int, default=16384)
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', action='store_true')
parser.add_argument('--lr_decay_steps', type=int, default=50000)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
parser.add_argument('--max_step', type=int, default=3000000)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--steps_per_print', type=int, default=100)
parser.add_argument('--steps_per_eval', type=int, default=1000)
parser.add_argument('--steps_per_visu', type=int, default=3456)
parser.add_argument('--epochs_per_save', type=int, default=5)
parser.add_argument('--visu_freq', type=int, default=10)
parser.add_argument('--store_grad', action='store_true')
parser.add_argument('--num_input_points', type=int, default=1024)
parser.add_argument('--dataset', default='modelnet40')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_input_points
NUM_GT_POINT = args.num_gt_points
DECAY_STEP = args.lr_decay_steps
DATASET = args.dataset

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def vary2fix(inputs, npts):
    inputs_ls = np.split(inputs[0], npts.cumsum())
    ret_inputs = np.zeros((1, BATCH_SIZE * NUM_POINT, 3), dtype=np.float32)
    ret_npts = npts.copy()
    for idx, obj in enumerate(inputs_ls[:-1]):
        if len(obj) <= NUM_POINT:
            select_idx = np.concatenate([
                np.arange(len(obj)), np.random.choice(len(obj), NUM_POINT - len(obj))])
        else:
            select_idx = np.arange(len(obj))
            np.random.shuffle(select_idx)
            pdb.set_trace()

        ret_inputs[0][idx * NUM_POINT:(idx + 1) * NUM_POINT] = obj[select_idx].copy()
        ret_npts[idx] = NUM_POINT
    return ret_inputs, ret_npts


def train(args):

    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')

    # for ModelNet, it is with Fixed Number of Input Points
    # for ShapeNet, it is with Varying Number of Input Points
    inputs_pl = tf.placeholder(tf.float32, (1, BATCH_SIZE * NUM_POINT, 3), 'inputs')
    npts_pl = tf.placeholder(tf.int32, (BATCH_SIZE,), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (BATCH_SIZE, args.num_gt_points, 3), 'ground_truths')
    add_train_summary('alpha', alpha)
    bn_decay = get_bn_decay(global_step)
    add_train_summary('bn_decay', bn_decay)

    model_module = importlib.import_module('.%s' % args.model_type, 'completion_models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha,
                               bn_decay=bn_decay, is_training=is_training_pl)

    # Another Solution instead of importlib:
    # ldic = locals()
    # exec('from completion_models.%s import Model' % args.model_type, globals(), ldic)
    # model = ldic['Model'](inputs_pl, npts_pl, gt_pl, alpha,
    # bn_decay=bn_decay, is_training=is_training_pl)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)
    saver = tf.train.Saver(max_to_keep=10)
    ''' from PCN paper:
    All our completion_models are trained using the Adam optimizer 
    with an initial learning rate of 0.0001 for 50 epochs
    and a batch size of 32. The learning rate is decayed by 0.7 every 50K iterations.
    '''

    if args.store_grad:
        grads_and_vars = trainer.compute_gradients(model.loss)
        for g, v in grads_and_vars:
            tf.summary.histogram(v.name, v, collections=['train_summary'])
            tf.summary.histogram(v.name + '_grad', g, collections=['train_summary'])

    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')

    # the input number of points for the partial observed data is not a fixed number
    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size,
        args.num_input_points, args.num_gt_points, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size,
        args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
    else:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y/n]' % args.log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "yes":
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        log.close()
        os.system('cp completion_models/%s.py %s' % (args.model_type, args.log_dir))  # bkp of model scripts
        os.system('cp train_completion.py %s' % args.log_dir)  # bkp of train procedure
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)  # GOOD habit

    log_fout = open(os.path.join(args.log_dir, 'log_train.txt'), 'a+')
    for arg in sorted(vars(args)):
        log_fout.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        log_fout.flush()

    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)

    for step in range(init_step + 1, args.max_step + 1):
        epoch = step * args.batch_size // num_train + 1
        ids, inputs, npts, gt = next(train_gen)
        if epoch > args.epoch:
            break
        if DATASET == 'shapenet8':
            inputs, npts = vary2fix(inputs, npts)

        start = time.time()
        feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: True}
        _, loss, summary = sess.run([train_op, model.loss, train_summary], feed_dict=feed_dict)
        total_time += time.time() - start
        writer.add_summary(summary, step)

        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0

        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            total_loss, total_time = 0, 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                _, inputs, npts, gt = next(valid_gen)
                if DATASET == 'shapenet8':
                    inputs, npts = vary2fix(inputs, npts)
                feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: False}
                loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)
                total_loss += loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0

        if step % args.steps_per_visu == 0:
            all_pcds = sess.run(model.visualize_ops, feed_dict=feed_dict)
            for i in range(0, args.batch_size, args.visu_freq):
                plot_path = os.path.join(args.log_dir, 'plots',
                                         'epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
                pcds = [x[i] for x in all_pcds]
                plot_pcd_three_views(plot_path, pcds, model.visualize_titles)

        if (epoch % args.epochs_per_save == 0) and \
                not os.path.exists(os.path.join(args.log_dir, 'model-%d.meta' % epoch)):
            saver.save(sess, os.path.join(args.log_dir, 'model'), epoch)
            print(colored('Epoch:%d, Model saved at %s' % (epoch, args.log_dir), 'white', 'on_blue'))

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':

    print('Now Using GPU:%s to train the model' % args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train(args)
