import argparse,logging
import numpy as np
import mxnet as mx

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(10, norm_stat)

def group(data, num_r, num, kernel, stride, pad, layer, down_sampling='pool'):
    if num_r > 0:
        conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1,1), name=('conv%s_r' % layer))
        slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
        act_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        conv = mx.symbol.Convolution(data=act_r, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    else:
        conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
    act = mx.symbol.maximum(slice[0], slice[1])
    if down_sampling == "pool":
        pool = mx.symbol.Pooling(data=act, pool_type="max", kernel=(2, 2), stride=(2,2), name=('pool%s' % layer))
    elif down_sampling == "conv":
        pool = mx.symbol.Convolution(data=act, kernel=(3,3), stride=(2,2), pad=(1,1), num_filter=num/2, name=('pool%s' % layer))
    return pool

def before_flatten(use_fuse=True):
    data = mx.symbol.Variable(name="data")
    pool1 = group(data, 0, 96, (5,5), (1,1), (2,2), str(1))
    pool2 = group(pool1, 96, 192, (3,3), (1,1), (1,1), str(2))
    pool3 = group(pool2, 192, 384, (3,3), (1,1), (1,1), str(3))
    pool4 = group(pool3, 384, 256, (3,3), (1,1), (1,1), str(4))
    pool5 = group(pool4, 256, 256, (3,3), (1,1), (1,1), str(5))
    if use_fuse:
        # # down-sampling pool2
        pool2_fuse_1 = mx.symbol.Convolution(data=pool2, kernel=(3,3), stride=(2,2), pad=(1,1), num_filter=384, name="pool2_fuse_1")
        pool2_fuse_1_slice = mx.symbol.SliceChannel(data=pool2_fuse_1, num_outputs=2, name="pool2_fuse_1_slice")
        pool2_fuse_1_act = mx.symbol.maximum(pool2_fuse_1_slice[0], pool2_fuse_1_slice[1])
        pool2_fuse_2 = mx.symbol.Convolution(data=pool2_fuse_1_act, kernel=(3,3), stride=(2,2), pad=(1,1), num_filter=256, name="pool2_fuse_2")
        pool2_fuse_2_slice = mx.symbol.SliceChannel(data=pool2_fuse_2, num_outputs=2, name="pool2_fuse_2_slice")
        pool2_fuse_2_act = mx.symbol.maximum(pool2_fuse_2_slice[0], pool2_fuse_2_slice[1])
        pool2_fuse = mx.symbol.Pooling(data=pool2_fuse_2_act, pool_type="max", kernel=(2, 2), stride=(2,2), name='pool2_fuse')
        # down-sampling pool3
        pool3_fuse_1 = mx.symbol.Convolution(data=pool3, kernel=(3,3), stride=(2,2), pad=(1,1), num_filter=256, name="pool3_fuse_1")
        pool3_fuse_1_slice = mx.symbol.SliceChannel(data=pool3_fuse_1, num_outputs=2, name="pool3_fuse_1_slice")
        pool3_fuse_1_act = mx.symbol.maximum(pool3_fuse_1_slice[0], pool3_fuse_1_slice[1])
        pool3_fuse = mx.symbol.Pooling(data=pool3_fuse_1_act, pool_type="max", kernel=(2, 2), stride=(2,2), name='pool3_fuse')
        # down-sampling pool4
        pool4_fuse_1 = mx.symbol.Convolution(data=pool4, kernel=(1,1), stride=(1,1), pad=(0,0), num_filter=256, name="pool4_fuse_1")
        pool4_fuse_1_slice = mx.symbol.SliceChannel(data=pool4_fuse_1, num_outputs=2, name="pool4_fuse_1_slice")
        pool4_fuse_1_act = mx.symbol.maximum(pool4_fuse_1_slice[0], pool4_fuse_1_slice[1])
        pool4_fuse = mx.symbol.Pooling(data=pool4_fuse_1_act, pool_type="max", kernel=(2, 2), stride=(2,2), name='pool4_fuse')
        return mx.symbol.Concat(*[mx.symbol.Flatten(data=pool5), mx.symbol.Flatten(data = pool4_fuse + pool3_fuse + pool2_fuse)])
    else:
        return mx.symbol.Flatten(data=pool5)

def lightened_moon_feature(num_classes=40, use_fuse=True):
    flatten = before_flatten(use_fuse)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
    fc1a = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    drop1 = mx.symbol.Dropout(data=fc1a, p=0.5, name="drop1")
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
    return fc2

def lightened_moon(num_classes=40, use_fuse=True):
    fc2 = lightened_moon_feature(num_classes=num_classes, use_fuse=use_fuse)
    moon = mx.symbol.MoonOutput(data=fc2, src_dist_path='./src_dict.txt', name='Moon')
    return moon

def main():
    # symbol = lightened_moon(num_classes=40, use_fuse=False)
    symbol = lightened_moon(num_classes=40, use_fuse=True)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = args.num_examples / args.batch_size
    checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
    kv = mx.kvstore.create(args.kv_store)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)
    train = mx.io.ImageRecordIter(
        path_imglist = args.list_dir + 'celeba_train.lst',
        path_imgrec = args.data_dir + "celeba_train.rec",
        label_width = 40,
        data_name   = 'data',
        label_name  = 'Moon_label',
        data_shape  = (1, 128, 128),
        scale       = 1./255,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    val = mx.io.ImageRecordIter(
        path_imglist = args.list_dir + 'celeba_val.lst',
        path_imgrec = args.data_dir + "celeba_val.rec",
        label_width = 40,
        data_name   = 'data',
        label_name  = 'Moon_label',
        batch_size  = args.batch_size,
        data_shape  = (1, 128, 128),
        scale       = 1./255,
        rand_crop   = True,
        rand_mirror = False,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = symbol,
        arg_params         = arg_params,
        aux_params         = aux_params,
        num_epoch          = 100,
        begin_epoch        = args.model_load_epoch,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.0005,
        lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=4*max(int(epoch_size * 1), 1), factor=0.8, stop_factor_lr=1e-5),
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))
    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = ['multi_binary_acc'],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 10),
        epoch_end_callback = checkpoint)
        # monitor            = mon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training lightened-moon")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./', help='the input data directory')
    parser.add_argument('--list-dir', type=str, default='/home/work/data/Face/CelebA/Img/img_celeba_cropped/',
                        help='the directory which contain the training list file')
    parser.add_argument('--model-save-prefix', type=str, default='../model/lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to save')
    parser.add_argument('--lr', type=float, default=0.05, help='initialization learning reate')
    parser.add_argument('--batch-size', type=int, default=384, help='the batch size')
    parser.add_argument('--num-examples', type=int, default=159923, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--model-load-prefix', type=str, default='../model/lightened_moon/lightened_moon', help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=0, help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    logging.info(args)
    main()

