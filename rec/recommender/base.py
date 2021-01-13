"""
Base module for Session-Aware Recommenders
"""
from abc import abstractmethod

from rec.base import ParametrizedObject
from rec.eval import PrecisionRecallAtN, MeanReciprocalRank


class SessionAwareRecommender(ParametrizedObject):
    @abstractmethod
    def fit(self, train_dataset, valid_data=None, valid_measures=None):
        """
        Method called in order to prepare model.
        Args:
            train_dataset: training dataset 

        Returns:

        """
        pass

    @abstractmethod
    def predict_single_session(self, session, n=10):
        """
        Give a prediction for a single user's session 
        Args:
            session (rec.model.Session): current user's session
            n (int): how many items to recommend

        Returns: (list[int]) list of recommended items ids 

        """
        pass

    def predict(self, sessions, n=20):
        return [self.predict_single_session(s, n) for s in sessions]

    def save_model(self, filename):
        """
        Serialize model to file. 
        Allows to save the trained model to the file and load it later with the
        `load_model()` method in order to reuse already trained model or just
         to do retraining. 
        Args:
            filename: file path for model serialization.

        Returns:

        """
        raise NotImplementedError()

    @staticmethod
    def load_model(filename):
        """
        Method loading the model from the before serialized file. 
        
        Args:
            filename: file path with saved model.

        Returns:

        """
        raise NotImplementedError()
