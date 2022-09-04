CUDA_VISIBLE_DEVICES=0, \
python3 run.py \
--musdb musdb18 \
--audio_channels 2 \
--batch_size 1 \
--logs demucs_pgd \
--test_pretrained demucs \
--evals demucs_pgd_wav \
--save \
--attack 

# python3 -m demucs \
# --musdb musdb18 \
# --audio_channels 2 \
# --batch_size 1 \
# --logs tasnet_pgd_test \
# --test_pretrained tasnet \
# --evals tasnet_pgd_test_wav \
# --save \
# --attack \
# -d cpu