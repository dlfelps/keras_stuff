import pickle
from keras.models import load_model, clone_model

from keras_pickle_wrapper import KerasPickleWrapper


class KerasArray:

    def __init__(self):
        self.models = []

    def copy_models(self, model, num_copies=10):
        #make copies
        self.models = [clone_model(model) for _ in range(num_copies)]

    def pickle(self, file_path):
        #wrap models in PickleWrapper
        self.models = [KerasPickleWrapper(m) for m in self.models]

        file = open(file_path, 'wb')
        pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

        #unwrap models so you don't change the state of this object
        self.models = [m.__call__() for m in self.models]

    @staticmethod
    def unpickle(file_path):
        file = open(file_path, 'rb')
        temp = pickle.load(file)
        file.close()

        #create new KerasArray object
        ka = KerasArray()

        #unwrap models
        ka.models = [m.__call__() for m in temp.models]
        return ka

if __name__ == "__main__":
    m = load_model('/home/dlfelps/PycharmProjects/untitled/saved_cnn.hdf5')
    ka = KerasArray()
    ka.copy_models(m, num_copies=2)

    ka.pickle('temp.pkl')
    ka2 = KerasArray.unpickle('temp.pkl')

