from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from joblib import dump

if __name__ == '__main__':
    """
    Train regressors for various text data
    coming from the Airbnb dataset
    """

    desc_tfidf = TfidfVectorizer(stop_words='english')
    neigh_tfidf = TfidfVectorizer(stop_words='english')
    host_tfidf = TfidfVectorizer(stop_words='english')

    X_train = pd.read_csv("./outputs/X_train.csv")
    X_dev = pd.read_csv("./outputs/X_dev.csv")

    y_train = pd.read_csv("./outputs/y_train.csv")
    y_dev = pd.read_csv("./outputs/y_dev.csv")

    desc_vec = desc_tfidf.fit_transform(X_train['description'].replace(np.nan, ' '))
    print('Transformed train description')
    desc_vec_test = desc_tfidf.transform(X_dev['description'].replace(np.nan, ' '))
    print('Transformed test description')
    desc_reg = LGBMRegressor(random_state=0).fit(desc_vec, y_train)
    X_train['desc_pred'] = desc_reg.predict(desc_vec)
    X_dev['desc_pred'] = desc_reg.predict(desc_vec_test)
    print('Trained desc regressor')

    neigh_vec = neigh_tfidf.fit_transform(X_train['neighborhood_overview'].replace(np.nan, ' '))
    print('Transformed train neighborhood')
    neigh_vec_test = neigh_tfidf.transform(X_dev['neighborhood_overview'].replace(np.nan, ' '))
    print('Transformed test neighborhood')
    neigh_reg = LGBMRegressor(random_state=0).fit(neigh_vec, y_train)
    X_train['neigh_pred'] = neigh_reg.predict(neigh_vec)
    X_dev['neigh_pred'] = neigh_reg.predict(neigh_vec_test)
    print('Trained neigh regressor')

    host_vec = host_tfidf.fit_transform(X_train['host_about'].replace(np.nan, ' '))
    print('Transformed train host')
    host_vec_test = host_tfidf.transform(X_dev['host_about'].replace(np.nan, ' '))
    print('Transformed test host')
    host_reg = LGBMRegressor(random_state=0).fit(host_vec, y_train)
    X_train['host_pred'] = host_reg.predict(host_vec)
    X_dev['host_pred'] = host_reg.predict(host_vec_test)
    print('Trained host regressor')

    dump(desc_reg, './models/desc_reg.joblib')
    dump(neigh_reg, './models/neigh_reg.joblib')
    dump(host_reg, './models/host_reg.joblib')
