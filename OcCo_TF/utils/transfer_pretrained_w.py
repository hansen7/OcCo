#  Copyright (c) 2020. Author: Hanchen Wang, hc.wang96@gmail.com

import os, argparse, tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from termcolor import colored

def load_para_from_saved_model(model_path, verbose=False):
    """load the all parameters from the saved TensorFlow checkpoint
    the format is dict -> {var_name(str): var_value(numpy array)}"""
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    var_to_map = reader.get_variable_to_shape_map()

    print('\n============================')
    print('model checkpoint: ', model_path)
    print('checkpoint has been loaded')
    for key in var_to_map.keys():
        var_to_map[key] = reader.get_tensor(key)
        if verbose:
            print('tensor_name:', key, ' shape:', reader.get_tensor(key).shape)
    print('============================\n')
    return var_to_map


def intersec_saved_var(model_path1, model_path2, verbose=False):
    """find the intersection of two saved models in terms of variable names"""
    var_to_map_1 = load_para_from_saved_model(model_path1, verbose=verbose)
    var_to_map_2 = load_para_from_saved_model(model_path2, verbose=verbose)

    # list of shared variable
    intersect = [*set(var_to_map_1.keys()).intersection(set(var_to_map_2.keys())), ]

    if verbose:
        print('\n=======================')
        print('the shared variables are:')
        print(intersect)

    return var_to_map_1, var_to_map_2, intersect


def load_pretrained_var(source_model_path, target_model_path, verbose=False):
    """save the parameters from source to target for variables in the intersection"""
    var_map_source, var_map_target, intersect = intersec_saved_var(
        source_model_path, target_model_path, verbose=verbose)

    out_f = open('para_restored.txt', 'w+')

    with tf.Session() as my_sess:
        new_var_list = []
        for var in var_map_target.keys():
            # pdb.set_trace()
            if (var in intersect) and (var_map_source[var].shape == var_map_target[var].shape):
                new_var = tf.Variable(var_map_source[var], name=var)
                if verbose:
                    print('%s has been restored from the pre-trained %s' % (var, source_model_path))
                out_f.writelines('Restored: %s has been restored from the pre-trained %s\n' % (var, source_model_path))
            else:
                new_var = tf.Variable(var_map_target[var], name=var)
                if verbose:
                    print('%s has been restored from the random initialized %s' % (var, target_model_path))
                out_f.writelines('Random Initialised: %s\n' % var)
            new_var_list.append(new_var)
        print('start to write the new checkpoint')
        my_sess.run(tf.global_variables_initializer())
        my_saver = tf.train.Saver(var_list=new_var_list)
        my_saver.save(my_sess, target_model_path)
        print(colored('source weights has been restored', 'white', 'on_blue'))

    my_sess.close()
    out_f.close()
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default='./pretrained/pcn_cd')
    parser.add_argument('--target_path', default='./log/pcn_cls_shapenet8_pretrained_init/model.ckpt')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    load_pretrained_var(args.source_path, args.target_path, args.verbose)
