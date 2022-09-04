CUDA_VISIBLE_DEVICES=0 \
python3 run.py \
--musdb musdb18 \
--audio_channels 2 \
--batch_size 1 \
--logs pgd \
--test_pretrained demucs \
--evals pgd_wav \
--attack \
--save