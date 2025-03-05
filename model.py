from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


class Model:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)

        print('accuracy_score', accuracy_score(y_test, y_pred))
        print('Confusion_matrix', confusion_matrix(y_test, y_pred))

    def get_weight(self):
        return self.model.coef_.flatten().tolist()

    def get_bias(self):
        return self.model.intercept_.tolist()

    def save_model(self):
        pickle.dump(self.model, open('model.pkl', 'wb'))

    def load_model(self):
        self.model = pickle.load(open('model.pkl', 'rb'))
