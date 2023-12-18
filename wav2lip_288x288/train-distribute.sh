#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://${MODEL_S3_BUCKET}/wav2lip_288x288/train_data/main2/* /tmp/main2/
rm -rf /tmp/main2/5540197286159433545/.ipynb_checkpoints
rm -rf /tmp/main2/5536876846942893978/.ipynb_checkpoints
pip install -r ./requirements.txt

echo "--------preprocess wav-----------"
python ./preprocess.py --data_root /tmp/main2/ --preprocessed_root /tmp/lrs2_preprocessed2/
echo "finished preprocess"

echo "--------prepare tain list--------"
python ./generate_filelists.py --data_root /tmp/lrs2_preprocessed2/
cat filelists/train.txt
echo "finished train list prepare"

echo "--------train the expert discriminator------"
startdt=$(date +%s)
deepspeed --num_gpus=8 ./color_syncnet_dist_train.py \
          --data_root /tmp/lrs2_preprocessed2/ \
          --checkpoint_dir /tmp/trained_syncnet/  \
          --deepspeed --deepspeed_config ds_config.json
enddt=$(date +%s)
interval_minutes=$(( (enddt - startdt) ))
echo "时间间隔为 $interval_minutes 秒"
echo "finished train syncnet"
#
#trained_syncnet_model_file=$(find /tmp/trained_syncnet/ -name  "*.pt" -print)
#trained_syncnet_model_file=$(echo $trained_syncnet_model_file|cut -d' ' -f1)
#echo "trained_syncnet_model_file: "${trained_syncnet_model_file}
    
#echo "--------train wav2lip-----------------------" 
#python ./hq_wav2lip_train_deepspeed.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_wav2lip_288x288/ --syncnet_checkpoint_path $trained_syncnet_model_file
#python ./hq_wav2lip_train.py --data_root /tmp/lrs2_preprocessed2/ --checkpoint_dir /tmp/trained_wav2lip_288x288/ --syncnet_checkpoint_path /tmp/trained_syncnet/10/global_step2/mp_rank_00_model_states.pt
          
#echo "finished wav2lip"

#./s5cmd sync /tmp/trained_wav2lip_288x288/ s3://$MODEL_S3_BUCKET/wav2lip_288x288/output/$(date +%Y-%m-%d-%H-%M-%S)/

###inference
#echo "begin inference"
#./s5cmd sync  s3://${sagemaker_default_bucket}/wav2lip/inference/face_video/* /tmp/face_video/
#./s5cmd sync  s3://${sagemaker_default_bucket}/wav2lip/inference/audio/* /tmp/audio/
#python ./inference.py --checkpoint_path /tmp/trained_wav2lip_288x288/checkpoint_step000000001.pth  --face /tmp/face_video/VID20230623143819.mp4 --audio /tmp/audio/测试wav2lip.mp3
#./s5cmd sync ./results/result_voice.mp4  s3://${sagemaker_default_bucket}/models/wav2lip_288x288/results/
