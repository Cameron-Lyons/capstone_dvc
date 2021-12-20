import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import shutil

if __name__ == '__main__':
    """
    Cleanup of data and train/test/dev split of data
    prior to use in model
    """

    df = pd.read_csv("./outputs/final_df.csv", low_memory=False)

    df['calendar_updated'] = pd.to_datetime(df['calendar_updated'])
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

    df = df.dropna(subset=['price','first_review','last_review','host_since'])

    data_2020 = [col for col in df.columns if '_0' in col]
    data_2019 = [col for col in df.columns if 'P00' in col or 'H00' in col]

    for col2020, col2019 in zip(data_2020, data_2019):
        df["delta_" + "col2020"] = df[col2020] - df[col2019]
        df["delta_" + "col2020"] = df[col2020] - df[col2019]


    X = df.drop(['price', 'id', 'listing_url', 'scrape_id',
                'last_scraped',
                'host_id', 'host_url', 'host_name',
                'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped',
                ], axis=1)

    X['calendar_updated'] = pd.to_numeric(pd.to_datetime(X['calendar_updated']))
    X['host_since'] = pd.to_numeric(pd.to_datetime(X['host_since']))

    X['first_review'] = pd.to_numeric(pd.to_datetime(X['first_review']))
    X['last_review'] = pd.to_numeric(pd.to_datetime(X['last_review']))

    y = df['price']
    y = y.str.replace('$', '', regex=False).str.replace('.', '', regex=False).str.replace(',','', regex=False).astype(int)  # convert to cents

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    for i, image_url in X_train["picture_url"].iteritems():
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            with open('./outputs/images/train/' + str(i) +'.jpg', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    
    for i, image_url in X_dev["picture_url"].iteritems():
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            with open('./outputs/images/dev/' + str(i) +'.jpg', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    
    for i, image_url in X_test["picture_url"].iteritems():
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            with open('./outputs/images/test' + str(i) +'.jpg', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f) 

    X_train.drop(['picture_url'], axis=1, inplace=True)
    X_dev.drop(['picture_url'], axis=1, inplace=True)
    X_test.drop(['picture_url'], axis=1, inplace=True)

    X_train.to_csv("./outputs/X_train.csv", index=False)
    X_dev.to_csv("./outputs/X_dev.csv", index=False)
    X_test.to_csv("./outputs/X_test.csv", index=False)

    y_train.to_csv("./outputs/y_train.csv", index=False)
    y_dev.to_csv("./outputs/y_dev.csv", index=False)
    y_test.to_csv("./outputs/y_test.csv", index=False)
    print(X_test.head())
    for i, image_url in X_train["picture_url"].iteritems():
        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            with open('./outputs/images/' + i, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
