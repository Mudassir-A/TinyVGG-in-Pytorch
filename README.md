# TinyVGG Model - Custom Dataset Implementation

## Instructions for Execution:
1. **Set Up a Virtual Environment**: 
   ```bash
   python -m venv .venv
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**: 
   Execute the `get_data.py` script to download the dataset into the `data/` directory:
   ```bash
   python get_data.py
   ```

4. **Train the Model**: 
   Use the `train.py` script to train and save a TinyVGG model on your dataset:
   ```bash
   python train.py
   ```
   Alternatively, you can specify parameters for training:
   ```bash
   python train.py --train_dir TRAIN_DIR --test_dir TEST_DIR --learning_rate LEARNING_RATE --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS 
   ```

5. **Make Predictions**: 
   Run the `predict.py` script to generate predictions on a specific image using the trained TinyVGG model:
   ```bash
   python predict.py --image IMAGE_PATH
   ```
   Here, `IMAGE_PATH` refers to the location of the image you wish to analyze.

### Example Usage:
```bash
python predict.py --image data/test/sushi/175783.jpg
```

This command will load the saved model and provide the predicted class along with the associated probability for the specified image.