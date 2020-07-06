import torch
import sys
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import classification_report
import importlib.util
import glob
from PIL import Image
from sklearn.metrics import confusion_matrix
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

## User Param
# Amazon
# args = {'dataset':'amazon'}
# args = {'dataset':'amazon',
#         'folder_name':'/home/yoyo/Desktop/SimCLR/runs/amazon_Jul05_23-00-54_iter200_yoyo',
#         }
# Oil Palm
args = {'dataset':'oilpalm',
        'folder_name':'/home/yoyo/Desktop/SimCLR/runs_oilpalm/Jul05_21-43-21_iter200_lr1e-4_yoyo',
        }

pickle_file = '/home/yoyo/Desktop/SimCLR/experiments/results/oilpalm_eval10.pkl'

DEBUG = False

print(args)

checkpoints_folder = os.path.join(args['folder_name'], 'checkpoints')
config = yaml.full_load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
print(config)


def _load_data(prefix="train", img_size=128):
    img_list = glob.glob('./data/satellite/' + args['dataset'] + '/' + prefix + '/**/*')
    np.random.shuffle(img_list)
    X_train = []
    for f in img_list:
        img = Image.open(f).convert('RGB').resize((img_size, img_size))
        X_train.append(img)
        img.load()
    X_train = np.stack(X_train).transpose(0,3,1,2)
    
    if args['dataset'] == 'amazon':
        tag = 'agri'
    else:
        tag = 'palm'
    y_train = np.array([f.split('/')[-2] == tag for f in img_list])
    
    print("===  {} dataset ===".format(args['dataset']))
    print("{} images".format(prefix))
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train


X_train, y_train = _load_data("train")

X_train_unlbl, y_train_unlbl = _load_data("unlabeled")
n = X_train_unlbl.shape[0]
n_spl = int(n*1.0)
X_train = np.concatenate((X_train, X_train_unlbl[:n_spl]))
y_train = np.concatenate((y_train, y_train_unlbl[:n_spl]))

if DEBUG:
    fig, axs = plt.subplots(nrows=2, ncols=6, constrained_layout=False, figsize=(12,4))

    for i, ax in enumerate(axs.flat):
        ax.imshow(X_train[i].transpose(1,2,0))
        ax.title.set_text(y_train[i])
    plt.show()


# load facemask test data
X_test, y_test = _load_data("test")

if DEBUG:
    fig, axs = plt.subplots(nrows=2, ncols=6, constrained_layout=False, figsize=(12,4))

    for i, ax in enumerate(axs.flat):
        ax.imshow(X_test[i].transpose(1,2,0))
        ax.title.set_text(y_test[i])
    plt.show()

args['x_train'] = X_train.shape
args['x_test'] = X_test.shape

def linear_model_eval(X_train, y_train, X_test, y_test):
    if args['dataset'] == 'amazon':
        tag = 'agri'
    else:
        tag = 'palm'
    
    clf = LogisticRegression(max_iter=1200, solver='lbfgs', C=10.0)
    clf.fit(X_train, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))
    predIdxs = clf.predict(X_test)
    # show a nicely formatted classification report
    logis_dict = classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3, output_dict=True)
    print(classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3))
    logis_dict['confusion_matrix'] = confusion_matrix(y_test, predIdxs)
    print(logis_dict['confusion_matrix'])
    
    print("-------------------------------")
    svm = SVC()
    svm.fit(X_train, y_train)
    print("SVM feature eval")
    print("Train score:", svm.score(X_train, y_train))
    print("Test score:", svm.score(X_test, y_test))
    predIdxs = svm.predict(X_test)
    # show a nicely formatted classification report
    svm_dict = classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3, output_dict=True)
    print(classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3))
    svm_dict['confusion_matrix'] = confusion_matrix(y_test, predIdxs)
    print(svm_dict['confusion_matrix'])
    
    print("-------------------------------")
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))
    predIdxs = neigh.predict(X_test)
    # show a nicely formatted classification report
    knn_dict = classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3, output_dict=True)
    print(classification_report(y_test, predIdxs, target_names=['no_'+tag, tag], digits=3))
    knn_dict['confusion_matrix'] = confusion_matrix(y_test, predIdxs)
    print(knn_dict['confusion_matrix'])
    
    with open(pickle_file,"wb") as f:
        pickle.dump([args, config, logis_dict, svm_dict, knn_dict],f)


def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        X_batch = torch.tensor(X[i: i+batch_size]) / 255.
        y_batch = torch.tensor(y[i: i+batch_size])
        yield X_batch.to(device), y_batch.to(device)

# Load the neural net module
spec = importlib.util.spec_from_file_location("model", os.path.join(checkpoints_folder, '../../../models/resnet_simclr.py'))
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)

model = resnet_module.ResNetSimCLR(**config['model'])
model.eval()

state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model = model.to(device)

X_train_feature = []

for batch_x, batch_y in next_batch(X_train, y_train, batch_size=config['batch_size']):
    features, _ = model(batch_x)
    X_train_feature.extend(features.cpu().detach().numpy())
    
X_train_feature = np.array(X_train_feature)

print("Train features")
print(X_train_feature.shape)

X_test_feature = []

for batch_x, batch_y in next_batch(X_test, y_test, batch_size=config['batch_size']):
    features, _ = model(batch_x)
    X_test_feature.extend(features.cpu().detach().numpy().reshape((-1,512)))
    
X_test_feature = np.array(X_test_feature)

print("Test features")
print(X_test_feature.shape)

args['X_train_feature'] = X_train_feature.shape
args['X_test_feature'] = X_test_feature.shape

scaler = preprocessing.StandardScaler()
scaler.fit(X_train_feature)

linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)