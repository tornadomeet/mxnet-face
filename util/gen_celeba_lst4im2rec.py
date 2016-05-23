import os, argparse, logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dictribute = [0]*40  # used to save src distribution
train_cnt = 0

def lst_write(file, cnt, anno_line, name):
    file.write(str(cnt) + '\t')
    for i in range(len(anno_line)-1):
        if(-1 == int(anno_line[i+1])):
            file.write('-1' + '\t')
        else:
            file.write(anno_line[i+1] + '\t')
    file.write(name+'.png\n')

def calc_dist(anno_line):
    global train_cnt, dictribute
    train_cnt += 1
    for i in range(len(anno_line)-1):
        if int(anno_line[i+1]) == 1:
            dictribute[i] += 1

def main():
    src_lines = open(args.src_list, 'r').readlines()
    partition_lines = open(args.partition, 'r').readlines()
    train_path = os.path.join(args.root, 'celeba_train.lst')
    val_path = os.path.join(args.root, 'celeba_val.lst')
    test_path = os.path.join(args.root, 'celeba_test.lst')
    anno_lines = open(args.anno, 'r').readlines()
    train_file = open(train_path, 'w')
    val_file = open(val_path, 'w')
    test_file = open(test_path, 'w')
    anno_idx = []
    for line in anno_lines[2:]:
        anno_idx.append(line.strip('\n').split()[0])
    cnt = non_cnt = 0
    for line in partition_lines:
        logging.info("processing {}".format(line.split()[0]))
        line = line.strip('\n').split()
        name = line[0].split('.')[0]
        flag = int(line[1])
        if name+'.png\n' in src_lines:
            index = anno_idx.index(name+'.jpg')
            anno_line = anno_lines[index+2].strip('\r\n').split()
            if flag == 0: # train
               lst_write(train_file, cnt, anno_line, name)
               calc_dist(anno_line)
            elif flag == 1: # val
                lst_write(val_file, cnt, anno_line, name)
            elif flag == 2:  # test
                lst_write(test_file, cnt, anno_line, name)
            else:
                logging.info("error partition, the number should be 0,1,2")
            cnt = cnt + 1
        else:
            non_cnt = non_cnt + 1
            logger.info("cannot found {} because face detection failed\n".format(name+'jpg'))
    logging.info("processed {} image, and {} images left un-processed".format(str(cnt), str(non_cnt)))
    train_file.close()
    val_file.close()
    test_file.close()
    if args.shuffle:
        cmd = "shuf " + train_path + " -o " + train_path
        os.system(cmd)
        cmd = "shuf " + val_path + " -o " + val_path
        os.system(cmd)
        cmd = "shuf " + test_path + " -o " + test_path
        os.system(cmd)
        logging.info("shuffle done!")
    logging.info("begin calc the src distribution, dictribute number is :\n {}".format(dictribute))
    with open('src_dict.txt', 'w') as f:
        for i in range(len(dictribute)-1):
            f.write(str(float(dictribute[i])/train_cnt)+" ")
        f.write(str(float(dictribute[-1])/train_cnt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate celeba_(train, val, test).lst for im2rec tool in mxnet')
    parser.add_argument('--root', type=str, default='/home/work/data/Face/CelebA/Img/img_celeba_cropped',
                        help='the root dir of image')
    parser.add_argument('--src-list', type=str, default='/home/work/data/Face/CelebA/Img/img_celeba_cropped/celeba.lst',
                        help='the src list file of CelebA cropped image')
    parser.add_argument('--partition', type=str, default='/home/work/data/Face/CelebA/Eval/list_eval_partition.txt',
                        help='the original partition file of CelebA image')
    parser.add_argument('--anno', type=str, default='/home/work/data/Face/CelebA/Anno/list_attr_celeba.txt',
                        help='40 attributes annotations file')
    parser.add_argument('--shuffle', action='store_true', default=False, help='true : shuffle the *.lst.')
    args = parser.parse_args()
    logging.info(args)
    main()