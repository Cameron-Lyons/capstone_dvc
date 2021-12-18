import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
import torch
import torchvision.transforms as transforms
from image_reg import abb_dataset, Net
from joblib import dump, load
from scripts.prep import X_dev, X_test, X_dev

desc_tfidf = TfidfVectorizer(stop_words='english')
neigh_tfidf = TfidfVectorizer(stop_words='english')
host_tfidf = TfidfVectorizer(stop_words='english')

desc_reg = load('models/desc_reg.joblib')
neigh_reg = load('models/neigh_reg.joblib')
host_reg = load('models/host_reg.joblib')

net = Net()
net.load_state_dict(torch.load('./models/cnn'))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

X_dev = pd.read_csv('outputs/X_dev.csv')
y_dev = pd.read_csv('outputs/y_dev.csv')

X_test = pd.read_csv('outputs/X_test.csv')
y_test = pd.read_csv('outputs/y_test.csv')

devset = abb_dataset(csv_file='./outputs/pics_dev.csv',
                    root='./outputs', train=False,
                    download=True, transform=transform)

devloader = torch.utils.data.DataLoader(devset, batch_size=4,
                                         shuffle=False, num_workers=2)

testset = abb_dataset(csv_file='./outputs/pics_test.csv',
                    root='./outputs', train=False,
                    download=True, transform=transform)

testloader = torch.utils.data.DataLoader(devset, batch_size=4,
                                         shuffle=False, num_workers=2)

X_dev['host_pred'] = net(devloader)
X_test['host_pred'] = net(testloader)

desc_vec = desc_tfidf.fit_transform(X_dev['description'].replace(np.nan, ' '))
print('Transformed train description')
desc_vec_test = desc_tfidf.transform(X_test['description'].replace(np.nan, ' '))
print('Transformed test description')
X_dev['desc_pred'] = desc_reg.predict(desc_vec)
X_test['desc_pred'] = desc_reg.predict(desc_vec_test)

neigh_vec = neigh_tfidf.fit_transform(X_dev['neighborhood_overview'].replace(np.nan, ' '))
print('Transformed train neighborhood')
neigh_vec_test = neigh_tfidf.transform(X_test['neighborhood_overview'].replace(np.nan, ' '))
print('Transformed test neighborhood')
X_dev['neigh_pred'] = neigh_reg.predict(neigh_vec)
X_test['neigh_pred'] = neigh_reg.predict(neigh_vec_test)

host_vec = host_tfidf.fit_transform(X_dev['host_about'].replace(np.nan, ' '))
print('Transformed train host')
host_vec_test = host_tfidf.transform(X_test['host_about'].replace(np.nan, ' '))
print('Transformed test host')
X_dev['host_pred'] = host_reg.predict(host_vec)
X_test['host_pred'] = host_reg.predict(host_vec_test)

X_dev.drop(['description', 'neighborhood_overview', 'host_about'], axis=1, inplace=True)
X_test.drop(['description', 'neighborhood_overview', 'host_about'], axis=1, inplace=True)

enc = OneHotEncoder(handle_unknown='ignore')

X_dev = enc.fit_transform(X_dev)
X_test = enc.transform(X_test)

reg = LGBMRegressor(random_state=0)
reg.fit(X_dev, y_dev)
print('Trained final model')
dump(reg, 'models/final_reg.joblib')