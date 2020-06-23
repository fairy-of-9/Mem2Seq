import pickle

def load_data(path):
    print('[INFO] load from {}'.format(path))
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)