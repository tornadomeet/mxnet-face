Using [MXNet](https://github.com/dmlc/mxnet) for Face-related Algorithm
-------------------------
-------------------------

About
--------
Using mxnet for face-related algorithm, here now only provide a trained lightened cnn[1] model together with the training script, the single model get *97.13%+-0.88%* accuracy on LFW, and with only 20MB size.

How to test
-----------
run ```./test.sh``` in shell.  
This script will run the evaluation on lfw using trained model, befor running, you should change your own ```align_data_path``` in test.sh.

How to train
------------
run ```./run.sh``` in shell.  
This script will train the lightened cnn face model, using [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset, more accurately, i used the [cleaned version](https://github.com/happynear/FaceVerification).  
Again, you should change with your own setting in run.sh, and using your own hyper-parameter when training the model.


Implemented details
----------------------
* you should installed the [dlib](https://github.com/davisking/dlib) and [opencv](https://github.com/Itseez/opencv) libirary with python interface firstly.
* using dlib for face detection and alignment like [openface](https://cmusatyalab.github.io/openface/), but you can also choose opencv for detection, i had provided the detection model in ```model/opencv/cascade.xml```.
* using my slightly changed mxnet branch [face](https://github.com/tornadomeet/mxnet/tree/face) for trainig.
* 385504 images for train, and 20290 for val.
* run ```./model/get-models.sh``` to download the ```shape_predictor_68_face_landmarks.dat``` for face alignment.

How to improve accuracy on LFW?
-------------------------------
* using more accurate aligned face images for trainig, currently the aligned face images for training has many mistake images, which will hurt the perfomance. you can using more powerful face detection and alignment for face processing.
* using more data
* add verification information
* ...

Reference
---------
[1] Wu X, He R, Sun Z. A Lightened CNN for Deep Face Representation[J]. arXiv preprint arXiv:1511.02683, 2015.  
