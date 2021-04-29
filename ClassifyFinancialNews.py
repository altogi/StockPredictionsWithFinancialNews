import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split


class FinancialNewsClassifier:
    def __init__(self, labeled_news, model='bert', max_len=500, validation_size=0.2, batch_size=32,
                 split_type='random', max_features=None):
        """
        This class acts as a wrapper for the preprocessing, definitions, training, and storing, related to a text
        classification model with ktrain. Given a dataset on which to train and test, and a set of parameters concerning
         preprocessing and training, a model predicting whether a financial news corresponds to an opportunity to buy or
         sell is defined. For the inputs:
         - labeled_news: This DataFrame contains not only the body of each financial news taken into account (in column
         'content', but a one-hot encoded label of every news, according to whether it is recommended to buy, sell, or
         do nothing (columns 'buy', 'sell', and 'do_nothing')
         - model: This string indicated what model should the text classifier be based on. According to  ktrain's
         source code, the available models are:
            'fasttext' for FastText model
            'nbsvm' for NBSVM model
            'logreg' for logistic regression using embedding layers
            'bigru' for Bidirectional GRU with pretrained word vectors
            'bert' for BERT Text Classification
            'distilbert' for Hugging Face DistilBert model
         - maxlen: This int indicates the maximum number of words that can be taken into account per document.
         - validation_size: This float sets the size of the validation set used to evaluate the model.
         - batch_size:
         - split_type: Two types of splitting into test and training sets are considered. split_type='random' simply
         randomly splits 'labeled_news' according to the specified fraction (validation_size), ensuring an homogeneous
         distribution of each class. split_type='time_series' ensures that only the latest entries of the dataset are
         taken as a validation set.
         - max_features: If model is neither 'bert' not 'distilbert', it is the maximum number of words to consider in
         the vocabulary during preprocessing.
         """

        self.data = labeled_news
        self.model_name = model
        self.max_len = max_len
        self.batch_size = batch_size
        if self.model_name not in ['bert', 'distilbert']:
            if max_features is None:
                self.max_features = 10000
            else:
                self.max_features = max_features

        # Split data into training and validation sets
        if split_type == 'random':
            self.data_train, self.data_validation = train_test_split(
                self.data[['id', 'content', 'buy', 'sell', 'do_nothing']],
                shuffle=True,
                test_size=validation_size,
                stratify=self.data[['buy', 'sell', 'do_nothing']])
        elif split_type == 'time_series':
            train_cut = int((1 - validation_size) * len(self.data))
            self.data_train = self.data \
                                  .sort_values(by=['release_date']) \
                                  .head(train_cut)[['id', 'content', 'buy', 'sell', 'do_nothing']]
            self.data_validation = self.data \
                                       .sort_values(by=['release_date']) \
                                       .tail(len(self.data) - train_cut)[['id', 'content', 'buy', 'sell', 'do_nothing']]

        self.data['is_validation'] = self.data['id'].isin(self.data_validation['id']).astype(int)

        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.train_preprocessed, self.test_preprocessed, self.preprocessing = None, None, None
        self.model, self.learner = None, None
        self.learning_rate = None
        self.predictor, self.predictions = None, None

    def preprocess_data(self):
        """This method preprocesses the split data, according to the specified model and parameters."""

        if self.model_name not in ['bert', 'distilbert']:
            preprocess_mode = 'standard'
        else:
            preprocess_mode = self.model_name

        self.train_preprocessed, self.test_preprocessed, self.preprocessing = text.texts_from_df(
            self.data_train,
            'content',
            label_columns=['buy', 'sell', 'do_nothing'],
            val_df=self.data_validation,
            maxlen=self.max_len,
            preprocess_mode=preprocess_mode,
            lang='en'
        )

    def define_model_and_learner(self):
        """Once the training and testing data have been preprocessed, a ktrain model and a learner can be defined."""

        self.model = text.text_classifier(
            self.model_name,
            self.train_preprocessed,
            preproc=self.preprocessing,
            multilabel=False
        )
        self.learner = ktrain.get_learner(
            self.model,
            train_data=self.train_preprocessed,
            val_data=self.test_preprocessed,
            batch_size=self.batch_size)

    def request_learning_rate(self, max_epochs=5):
        """Before training, this method helps the user in order to set an appropriate reference learning rate."""
        self.learner.lr_find(max_epochs=max_epochs)
        self.learner.lr_plot()

    def train(self, lr, cycles=10, early_stopping=3, directory='./confusion_matrix.csv'):
        """This method trains the model based on a user-specified learning rate"""
        self.learning_rate = lr
        self.learner.autofit(self.learning_rate, epochs=cycles, early_stopping=early_stopping)
        self.learner.plot()
        self.learner.validate(print_report=False, save_path=directory)

    def get_predictor(self, directory):
        """This method obtains the predictor from the trained model, saves it, and uses it to carry out a new set of
        predictions."""
        self.predictor = ktrain.get_predictor(self.learner.model, self.preprocessing)
        self.predictor.save(directory + '/predictor')
        self.predictions = self.data[['id', 'content', 'buy', 'sell', 'do_nothing', 'is_validation']].copy()
        self.predictions['prediction'] = self.predictor.predict(self.predictions['content'].tolist())
