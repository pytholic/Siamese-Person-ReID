# Install

## Using pip

Install requirements.

```
pip install -r requirements.txt
```

## Docker

Alternatively use docker image.

Build the image.

```
make build-image
```

Run the container.

```
make run
```

Make sure to change the mount volume paths according to your system in `Makefile`.

# Train

Run `train.py`

```
python train.py --learning_rate <learning rate> --batch_size <size> --epochs <number of epochs>
```

# Datasets
Currently supported datasets.
* Market1501
* MSMT17

# Backbones
* mynet
* osnet
* timm models

Further backbone implementation can be found in my repository.
