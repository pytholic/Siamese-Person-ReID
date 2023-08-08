# Run

## Using pip

Install requirements.

```
pip install -r requirements.txt
```

Run `train.py`

```
python train.py --learning_rate <learning rate> --batch_size <size> --epochs <number of epochs>
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
