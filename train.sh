#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync  s3://${sagemaker_default_bucket}/wav2lip/train_data/main2/* /tmp/main2/
pip install -r ./requirements.txt

##preprocess wav
cd ./Wav2Lip && python ./preprocess.py --data_root /tmp/main2 --preprocessed_root /tmp/lrs2_preprocessed2/
find /tmp/lrs2_preprocessed2/ -name "*.wav"

###train the expert discriminator
python ./color_syncnet_train.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_syncnet/
###train wav2lip 
python ./hq_wav2lip_train.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_wav2lip/ --syncnet_checkpoint_path /tmp/trained_syncnet/

./s5cmd sync /tmp/trained_wav2lip/ s3://${sagemaker_default_bucket}/models/wav2lip/output/$(date +%Y-%m-%d-%H-%M-%S)/
