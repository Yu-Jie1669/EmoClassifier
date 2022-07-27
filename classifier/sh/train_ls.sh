# label smoothing
python ../bin/train.py --output=../../train/model-ls.pkl --device=2 --batch_size=40 --smoothing=0.1 > ../../train/model-ls.log