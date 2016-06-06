import argparse,logging
import numpy as np
import mxnet as mx
import cv2, dlib
from lightened_moon import lightened_moon_feature

logger = logging.getLogger()
logger.setLevel(logging.INFO)
import pdb

def main():
    symbol = lightened_moon_feature(num_classes=40, use_fuse=True)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    _, arg_params, aux_params = mx.model.load_checkpoint(args.model_load_prefix, args.model_load_epoch)
    detector = dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier(args.opencv)
    # read img and drat face rect
    img = cv2.imread(args.img)
    faces = detector(img, 1)
    gray = np.zeros(img.shape[0:2])
    if len(faces) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        opencv_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in opencv_faces:
            faces.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
    max_face = faces[0]
    if len(faces) > 0:
            max_face = max(faces, key=lambda rect: rect.width() * rect.height())
    for f in faces:
        if f == max_face:
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
        else:
            cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (255,0,0), 2)
    cv2.imwrite(args.img.replace('jpg', 'png'), img)
    # crop face area
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pad = [0.25, 0.25, 0.25, 0.25] if args.pad is None else args.pad
    left = int(max(0, max_face.left() - max_face.width()*float(pad[0])))
    top = int(max(0, max_face.top() - max_face.height()*float(pad[1])))
    right = int(min(gray.shape[1], max_face.right() + max_face.width()*float(pad[2])))
    bottom = int(min(gray.shape[0], max_face.bottom()+max_face.height()*float(pad[3])))
    gray = gray[left:right, top:bottom]
    gray = cv2.resize(gray, (args.size, args.size))/255.0
    img = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)
    # get pred
    arg_params['data'] = mx.nd.array(img, devs)
    exector = symbol.bind(devs, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exector.forward(is_train=False)
    exector.outputs[0].wait_to_read()
    output = exector.outputs[0].asnumpy()
    text = ["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald", "Bangs","Big_Lips","Big_Nose",
            "Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee",
            "Gray_Hair", "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
            "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
            "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
    pred = np.ones(40)
    print("attribution is:")
    for i in range(40):
        print text[i].rjust(20)+" : \t",
        if output[0][i] < 0:
            pred[i] = -1
            print "No"
        else:
            pred[i] = 1
            print "Yes"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict the face attribution of one input image")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--img', type=str, default='./1.jpg', help='the input img path')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--opencv', type=str, default='../model/opencv/cascade.xml',
                        help='the opencv model path')
    parser.add_argument('--pad', type=float, nargs='+',
                                 help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--model-load-prefix', type=str, default='../model/lightened_moon/lightened_moon_fuse',
                        help='the prefix of the model to load')
    parser.add_argument('--model-load-epoch', type=int, default=82,
                        help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()
    logging.info(args)
    main()

