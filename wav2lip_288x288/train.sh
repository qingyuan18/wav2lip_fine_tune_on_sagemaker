#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://${sagemaker_default_bucket}/wav2lip_288x288/train_data/main2/* /tmp/main2/
pip install -r ./requirements.txt

echo "--------preprocess wav-----------"
python ./preprocess.py --data_root /tmp/main2 --preprocessed_root /tmp/lrs2_preprocessed2/
echo "finished preprocess"

echo "--------prepare tain list--------"
python ./generate_filelists.py --data_root /tmp/lrs2_preprocessed2/
echo "finished train list prepare"

echo "--------train the expert discriminator------"
startdt=$(date +%s)
python ./color_syncnet_train.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_syncnet/
enddt=$(date +%s)
interval_minutes=$(( (enddt - startdt) ))
echo "时间间隔为 $interval_minutes 秒"
echo "finished train syncnet"


echo "--------train wav2lip-----------------------" 
python ./hq_wav2lip_train.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_wav2lip_288x288/ --syncnet_checkpoint_path /tmp/trained_syncnet/checkpoint_step000000001.pth
echo "finished wav2lip"

./s5cmd sync /tmp/trained_wav2lip_288x288/ s3://${sagemaker_default_bucket}/models/wav2lip_288x288/output/$(date +%Y-%m-%d-%H-%M-%S)/

###inference
#echo "begin inference"
#./s5cmd sync  s3://${sagemaker_default_bucket}/wav2lip/inference/face_video/* /tmp/face_video/
#./s5cmd sync  s3://${sagemaker_default_bucket}/wav2lip/inference/audio/* /tmp/audio/
#python ./inference.py --checkpoint_path /tmp/trained_wav2lip_288x288/checkpoint_step000000001.pth  --face /tmp/face_video/VID20230623143819.mp4 --audio /tmp/audio/测试wav2lip.mp3
#./s5cmd sync ./results/result_voice.mp4  s3://${sagemaker_default_bucket}/models/wav2lip_288x288/results/
