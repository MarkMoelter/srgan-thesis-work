LR_TO_TEST = [16, 32, 64]
HR_TO_TEST = [64, 128, 256, 512]
SCALE_FACTOR = [2, 4, 8]

# Dataset generation
IMG_DIR = 'data_gen/master_dataset'
LABEL_PATH = 'data_gen/image_labels.csv'

PARAM_PATH = 'Results/params'
GEN_PARAM_FILENAME = 'gen_params.pth'
DISC_PARAM_FILENAME = 'disc_params.pth'

# Training
EPOCH_NUM = 5
TRAIN_DATA_PATH = 'Results/dataset/training_set/BlurRad_0'
TRAIN_CSV_PATH = 'Results/CSVs/training'

# Validation
VAL_DATA_PATH = 'Results/dataset/testing_set'
IMG_DIR = 'Results/Images'
VAL_CSV_PATH = 'Results/CSVs/testing/metrics.csv'

# Accuracy
PYTESSERACT_ENV_VAR_PATH = r'C:\Program Files\Tesseract-OCR\tesseract'
ACC_CSV_PATH = 'Results/CSVs/recog&acc/Final_Accuracy.csv'
