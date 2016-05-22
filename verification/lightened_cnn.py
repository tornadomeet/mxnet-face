import argparse,logging
import mxnet as mx

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)

def group(data, num_r, num, kernel, stride, pad, layer):
    if num_r > 0:
        conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1,1), name=('conv%s_r' % layer))
        slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
        mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        conv = mx.symbol.Convolution(data=mfm_r, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    else:
        conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
    mfm = mx.symbol.maximum(slice[0], slice[1])
    pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(2, 2), stride=(2,2), name=('pool%s' % layer))
    return pool

def lightened_cnn_a_feature():
    data = mx.symbol.Variable(name="data")
    pool1 = group(data, 0, 96, (9,9), (1,1), (0,0), str(1))
    pool2 = group(pool1, 0, 192, (5,5), (1,1), (0,0), str(2))
    pool3 = group(pool2, 0, 256, (5,5), (1,1), (0,0), str(3))
    pool4 = group(pool3, 0, 384, (4,4), (1,1), (0,0), str(4))
    flatten = mx.symbol.Flatten(data=pool4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
    mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
    return drop1

def lightened_cnn_a(num_classes=10575):
     drop1 = lightened_cnn_a_feature()
     fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
     softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
     return softmax

def lightened_cnn_b_feature():
     data = mx.symbol.Variable(name="data")
     pool1 = group(data, 0, 96, (5,5), (1,1), (2,2), str(1))
     pool2 = group(pool1, 96, 192, (3,3), (1,1), (1,1), str(2))
     pool3 = group(pool2, 192, 384, (3,3), (1,1), (1,1), str(3))
     pool4 = group(pool3, 384, 256, (3,3), (1,1), (1,1), str(4))
     pool5 = group(pool4, 256, 256, (3,3), (1,1), (1,1), str(5))
     flatten = mx.symbol.Flatten(data=pool5)
     fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
     slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
     mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
     drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
     return drop1

def lightened_cnn_b(num_classes=10575):
     drop1 = lightened_cnn_b_feature()
     fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
     softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
     return softmax

def main():
    # lightened_cnn = lightened_cnn_a()
    lightened_cnn = lightened_cnn_b()
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = args.num_examples / args.batch_size
    checkpoint = mx.callback.do_checkpoint(args.model_save_prefix)
    kv = mx.kvstore.create(args.kv_store)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)

    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "casia_train.rec",
        data_shape  = (1, 128, 128),
        scale       = 1./255,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    if not args.retrain:
        val = mx.io.ImageRecordIter(
            path_imgrec = args.data_dir + "casia_val.rec",
            batch_size  = args.batch_size,
            data_shape  = (1, 128, 128),
            scale       = 1./255,
            rand_crop   = True,
            rand_mirror = False,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)
    else:
        val = None
    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = lightened_cnn,
        arg_params         = arg_params,
        aux_params         = aux_params,
        num_epoch          = 200,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.0005,
        lr_scheduler       = mx.lr_scheduler.FactorScheduler(step=5*max(int(epoch_size * 1), 1), factor=0.8, stop_factor_lr=5e-5),
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))
    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 100),
        epoch_end_callback = checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training lightened-cnn")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./', help='the input data directory')
    parser.add_argument('--model-save-prefix', type=str, default='../model/lightened_cnn/lightened_cnn',
                        help='the prefix of the model to save')
    parser.add_argument('--lr', type=float, default=0.05, help='initialization learning reate')
    parser.add_argument('--batch-size', type=int, default=384, help='the batch size')
    parser.add_argument('--num-examples', type=int, default=385504, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--model-load-prefix', type=str, default='../model/lightened_cnn', help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=1, help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    logging.info(args)
    main()

