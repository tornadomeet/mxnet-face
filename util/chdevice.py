import argparse
import mxnet as mx
import logging
import pdb

def ch_dev(arg_params, aux_params, ctx):
    for k, v in arg_params.items():
        arg_params[k] = mx.nd.array(v.asnumpy(), ctx=ctx)
    for k, v in aux_params.items():
        aux_params[k] = mx.nd.array(v.asnumpy(), ctx=ctx)
    return arg_params, aux_params

def main():
    ctx = mx.cpu()
    if args.dev >= 0:
        ctx = mx.gpu(0)
    else:
        assert(args.dev == -1)

    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    save_callback = mx.callback.do_checkpoint(args.prefix+"_new")
    save_callback(args.epoch-1, symbol, arg_params, aux_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='change the model on saved on a specific device, used in mxnet')
    parser.add_argument('--prefix', default='../model/lightened_cnn/lightened_cnn',
                        help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=166,
                        help='The epoch number of fcn16s model.')
    parser.add_argument('--dev', type=int, default=-1,
                        help='the task dev which you want the model on, -1:cpu, 0~N:gpu(0)')
    args = parser.parse_args()
    logging.info(args)
    main()
