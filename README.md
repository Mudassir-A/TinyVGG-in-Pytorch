# TinyVGG Model - on custom dataset

## How to run:
- Create a virtual environment: 
``` bash
pythom -m venv .venv
```
- Install requirements
``` bash
pip install -r requirements.txt
```
- run `get_data.py` to download the data into the `data\` directory
``` bash
python get_data.py
```
- run the `train.py` file to train and save a TinyVGG model on the given dataset
``` bash
python train.py
```
OR
``` bash
python train.py --train_dir TRAIN_DIR --test_dir TEST_DIR --learning_rate LEARNING_RATE --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS 
```