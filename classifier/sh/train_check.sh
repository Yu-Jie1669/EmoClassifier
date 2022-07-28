# load checkpoint disltillation
python ../bin/train.py --output=../../train/model-check-diatillation.pkl --device=3 --batch_size=32 --smoothing=0.1 --checkpoint=../../train/gs3907.pkl >../../train/model-check-diatillation.log
