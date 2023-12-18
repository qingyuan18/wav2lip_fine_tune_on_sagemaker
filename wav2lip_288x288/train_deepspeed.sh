mkdir -p ./savedmodel

deepspeed --num_gpus=4 ./color_syncnet_train_deepspeed.py \
          --data_root ../LSR2/lrs2_preprocessed_288x288/ \
          --checkpoint_dir ./savedmodel \
          --deepspeed --deepspeed_config ds_config.json
