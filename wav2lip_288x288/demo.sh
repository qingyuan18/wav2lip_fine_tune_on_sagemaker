python inference.py \
    --checkpoint_path ./savedmodel/wav2lip_checkpoint_step000001000.pth \
    --face ../videos/97.mp4 \
    --audio ../videos/test.wav \
    --outfile results/result_wav2lip.mp4 \
    --resize_factor 2
