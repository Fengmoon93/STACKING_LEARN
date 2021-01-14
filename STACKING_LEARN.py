import numpy as np
import copy
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import GlobalPrameters as g
class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.testSize=0.3
        self.k = 5
        self.basicModels=[]
        self.metaModel=None
    def SaveModel(self):
        if not os.path.exists(g.modelSavePath):
            os.makedirs(g.modelSavePath)
        for id,clf in enumerate(self.basicModels):
            joblib.dump(clf, g.modelSavePath+"/basic_"+str(id)+".pkl")
        joblib.dump(self.metaModel,g.modelSavePath+"/meta.pkl")
    def load_data(self):
        #prepare basic data
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.testSize, random_state=23)
        # print(self.x_test.shape,self.y_train.shape)
    def StackingModel(self,basicLearners=None,metaModel=None):
        #set basicModels
        if len(basicLearners)==0 or metaModel is None:
            raise ValueError("StackingModel Constructing Failed...")
        train_meta_model = None
        test_meta_model = None
        # Start stacking
        for clf_id, clf in basicLearners:
            def SaveBasicModel(clf):
                res = copy.deepcopy(clf)
                res.fit(self.x_train, self.y_train)
                return res
            self.basicModels.append(SaveBasicModel(clf))
            # Predictions for each classifier based on k-fold
            [predictions_clf,test_out0] = self.k_fold_cross_validation(clf)
            # Stack predictions which will form
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf
            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, test_out0))
            else:
                test_meta_model = test_out0
        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Train metaModel
        self.train_level_1(metaModel, train_meta_model, test_meta_model)

    def k_fold_cross_validation(self, clf):

        predictions_clf = None
        predictions_clf_test = None
        # Number of samples per fold
        batch_size = int(len(self.x_train) / self.k)

        # Stars k-fold cross validation
        for fold in range(self.k):

            # Settings for each batch_size
            if fold == (self.k - 1):
                test = self.x_train[(batch_size * fold):, :]
                batch_start = batch_size * fold
                batch_finish = self.x_train.shape[0]
            else:
                test = self.x_train[(batch_size * fold): (batch_size * (fold + 1)), :]
                batch_start = batch_size * fold
                batch_finish = batch_size * (fold + 1)

            # test & training samples for each fold iteration
            fold_x_test = self.x_train[batch_start:batch_finish, :]
            fold_x_train = self.x_train[[index for index in range(self.x_train.shape[0]) if
                                         index not in range(batch_start, batch_finish)], :]

            # test & training targets for each fold iteration
            fold_y_test = self.y_train[batch_start:batch_finish]
            fold_y_train = self.y_train[
                [index for index in range(self.x_train.shape[0]) if index not in range(batch_start, batch_finish)]]

            # Fit current classifier
            clf.fit(fold_x_train, fold_y_train)
            fold_y_pred = clf.predict(fold_x_test)

            # Store predictions for each fold_x_test
            if isinstance(predictions_clf, np.ndarray):
                predictions_clf = np.concatenate((predictions_clf, fold_y_pred))
            else:
                predictions_clf = fold_y_pred

            # Generate predictions for x_test
            y_pred_test = clf.predict(self.x_test)
            # Store predictions for each fold_x_test
            if isinstance(predictions_clf_test, np.ndarray):
                predictions_clf_test = np.vstack((predictions_clf_test, y_pred_test))
            else:
                predictions_clf_test = y_pred_test
        test_out=np.mean(predictions_clf_test.T,axis=1)
        return [predictions_clf,test_out]


    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.y_train)
        self.metaModel=final_learner
        # Getting train and test accuracies from meta_model
        print("Train accuracy:",final_learner.score(train_meta_model, self.y_train))
        print("Test accuracy:", final_learner.score(test_meta_model, self.y_test))
        self.SaveModel()

    def predict(self,input):
        predictions = None
        for basicModel in self.basicModels:
            y_pred=basicModel.predict(input)
            if isinstance(predictions, np.ndarray):
                predictions = np.vstack((predictions, y_pred))
            else:
                predictions = y_pred
        predictions=predictions.T
        out = self.metaModel.predict(predictions)
        return out
if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    basicLearners = [('dt', DecisionTreeClassifier()),
                     ('knn', KNeighborsClassifier()),
                     ('rf', RandomForestClassifier()),
                     ('gb', GradientBoostingClassifier()),
                     ('gn', GaussianNB())]
    final_learner = LogisticRegression()
    ensemble.StackingModel(basicLearners,final_learner)