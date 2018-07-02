# Experiments with Adam

This repo contains the scripts used to perfom the experiments in [this blog post](http://www.fast.ai/2018/07/02/adam-weight-decay/). If you're using this code or out results, please cite it appropriately. You will find
- a script to train [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) to >94% accuracy in 30 epochs without Test Time Augmentation or 18 with.
- a script to finetune a pretrained resnet50 on the [Standford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) to 90% accuracy in 60 epochs.
- a script to train an AWD LSTM (or QRNN) to the same perplexity as [the Saleforce team](https://github.com/salesforce/awd-lstm-lm) that created them but in just 90 epochs.

## Requirements

- the [fastai library](https://github.com/fastai/fastai) is necessary to run all the models. If you didn't pip-install it, don't forget to have a simlink pointing to it in the directory where you clone this repo.
- additonaly, the QRNNs model requires the [cupy library](https://github.com/cupy/cupy)

## To install

Run the script get_data.sh that will download and organize the data needed for the models

## Experiments

### Cifar10 dataset

- this should train cifar10 to 94.25% accuracy (on average):
```
python train_cifar10.py 3e-3 --wd=0.1 --wd_loss=False
```
- this should train cifar10 to 94% accuracy (on average) but faster:
```
python train_cifar10.py 3e-3 --wd=0.1 --wd_loss=False --cyc_len=18 --tta=True
```

### Stanford cars dataset

- this should get 90% accuracy (on average) without TTA, 91% with:
```
python fit_stanford_cars.py '(1e-2,3e-3)' --wd=1e-3 --tta=True
```

### Language models

- this should train an AWD LSTM to 68.7/65.5 perplexity without cache pointer, 52.9/50.9 with
```
python train_rnn.py 5e-3 --wd=1.2e-6 --alpha=3 --beta=1.5
```

- this should train an AWD QRNN to 69.6/66.7 perplexity without cache pointer, 53.6/51.7 with
```
python train_rnn.py 5e-3 --wd=1e-6 --qrnn=True
```


