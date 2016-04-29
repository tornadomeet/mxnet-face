#!/usr/bin/env sh
#
# Download model, this script refer to openface:https://github.com/cmusatyalab/openface

mkdir -p dlib
if [ ! -f dlib/shape_predictor_68_face_landmarks.dat ]; then
  printf "\n\n====================================================\n"
  printf "Downloading dlib's public domain face landmarks model.\n"
  printf "Reference: https://github.com/davisking/dlib-models\n\n"
  printf "This will incur about 60MB of network traffic for the compressed\n"
  printf "models that will decpmoress to about 100MB on disk.\n"
  printf "====================================================\n\n"
  wget -nv http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    -O dlib/shape_predictor_68_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error in wget."
  bunzip2 dlib/shape_predictor_68_face_landmarks.dat.bz2
  [ $? -eq 0 ] || die "+ Error using bunzip2."
fi
echo "download dlib/shape_predictor_68_face_landmarks done!"
