# CUDA_VISIBLE_DEVICES=1, \
# python3 -m  demucs \
# --musdb musdb18 \
# --audio_channels 2 \
# --batch_size 4 \
# --logs conv_tas \
# --tasnet \

CUDA_VISIBLE_DEVICES=1, \
python3 -m  demucs.run \
--musdb musdb18 \
--audio_channels 2 \
--batch_size 4 \
--logs conv_tas \
--tasnet \
