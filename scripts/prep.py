import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    """
    Cleanup of data and train/test/dev split of data
    prior to use in model
    """

    df = pd.read_csv("../outputs/final_df.csv", low_memory=False)

    df['calendar_updated'] = pd.to_datetime(df['calendar_updated'])
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')

    df = df.dropna(subset=['price','first_review','last_review','host_since'])

    df_head = df.head(20)
    df_head.to_csv('../outputs/example_df.csv')

    X = df.drop(['price', 'id', 'listing_url', 'scrape_id',
                'last_scraped',
                'picture_url', 'host_id', 'host_url', 'host_name',
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

    X_train.to_csv("../outputs/X_train.csv", index=False)
    X_dev.to_csv("../outputs/X_dev.csv", index=False)
    X_test.to_csv("../outputs/X_test.csv", index=False)

    y_train.to_csv("../outputs/y_train.csv", index=False)
    y_dev.to_csv("../outputs/y_dev.csv", index=False)
    y_test.to_csv("../outputs/y_test.csv", index=False)
