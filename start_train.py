import warnings

from preprocess import DataPreprocessor
import train
warnings.filterwarnings("ignore")

from utils import *
from config import *

data_train = DataPreprocessor()
urls_train, labels_train = read_data(INPUT_PATH_TRAIN)
data_train.do_preprocess(urls_train, labels_train)
train = train.TrainNet()
train.do_train_net(data_train)
