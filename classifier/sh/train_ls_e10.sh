# label smoothing epoch 10
python ../bin/train.py --output=../../train/model-ls.pkl --device=2 --batch_size=32 --smoothing=0.1 --epoch=10 > ../../train/model-ls_e10.log