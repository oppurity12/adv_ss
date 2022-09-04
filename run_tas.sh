CUDA_VISIBLE_DEVICES=1,2,3,4 \
python3 run.py \
--musdb musdb18 \
--audio_channels 2 \
--batch_size 12 \
--logs conv_tas \
--tasnet \
--X=10 \
--samples=80000 \
--epochs=180 \

