import pandas as pd
import datetime
from pathlib import Path
import ktrain
import pickle
import matplotlib.pyplot as plt
from ImportFinancialData import FinancialDataImporter
from LabelFinancialData import FinancialDataLabeler
from ClassifyFinancialNews import FinancialNewsClassifier
from SimulatePortfolio import PortfolioSimulator

pd.options.mode.chained_assignment = None


class FinancialNewsPredictor:
    """This is the main class of the project StockPredictionsWithFinancialNews. It acts as a wrapper of all of the
    project's scripts, in order to execute all of them in a simple manner with a few lines of code. It makes sure that
    the results generated in previous steps are properly saved so that further processing can easily load the results
    knowing the chosen parameters of the case. Regarding its input:
    - data: pandas DataFrame containing a series of financial news, where each row includes a field 'ticker' for the
    ticker of the security mentioned in each news, a field 'release_date' representing the date of the news, an
    identification field 'id', and a text field 'content' containing the news of every row.
    -base_directory: Base directory in which to store all data.
    -selection: List of tickers, sectors, industries, to select from the overall dataset. If selecting all entries,
    leave as default.
    -selection_mode: String that specifies selection mode. 'sector' selects entries by sector, 'industry' selects
    entries by industry, 'ticker' selects entries by tickers, and None selects all entries."""

    def __init__(self, data, base_directory='./', selection=None, selection_mode=None):
        self.data = data
        self.directory_main = base_directory

        if type(selection) == list and sum([type(s) != str for s in selection]) == 0:
            self.selection = selection
        elif type(selection) == str:
            self.selection = list(selection)
        else:
            self.selection = None

        if type(selection_mode) == str:
            if selection_mode in ['sector', 'industry', 'ticker'] and self.selection is not None:
                self.selection_mode = selection_mode
            else:
                self.selection_mode = None
        else:
            self.selection_mode = None

        if type(self.data['release_date'].values[0]) != datetime.datetime:
            self.data['release_date'] = pd.to_datetime(self.data['release_date'])
        self.ticker_dates = self.data[['id', 'ticker', 'release_date']]

        self.deltas = None
        self.market_data = None
        self.data_importer = None
        self.data_labeler = None
        self.classifier_trainer = None
        self.predictions = None
        self.model = None
        self.directory_selected, self.directory_labeled, self.directory_model = None, None, None
        self.directory_portfolio, self.simulator, self.simulated_portfolio = None, None, None

    def import_financial_data(self, deltas=None):
        """This method creates an instance of class FinancialDataImporter in order to import market prices of the
        companies in the original dataset."""

        if deltas is None:
            deltas = [2, 3, 5, 10, 15]
        self.deltas = deltas

        # Create a folder for this selection of deltas
        self.directory_main = self.directory_main + 'deltas=' + ','.join([str(d) for d in self.deltas])
        Path(self.directory_main).mkdir(parents=True, exist_ok=True)

        # Before importing, see if the file to generate already exists
        if Path(self.directory_main + '/market_data.csv').is_file():
            self.market_data = pd.read_csv(self.directory_main + '/market_data.csv', sep='|')
        else:
            self.data_importer = FinancialDataImporter(self.ticker_dates, deltas=self.deltas)
            self.data_importer.download_prices()
            self.data_importer.insert_prices()
            self.market_data = self.data_importer.df
            self.market_data.to_csv(self.directory_main + '/market_data.csv', sep='|')

    def apply_selection(self):
        """This method applies the selection of the dataset specified by the user"""
        if self.selection is not None:
            self.directory_selected = \
                self.directory_main + '/' + self.selection_mode + '=[' + ','.join([s for s in self.selection]) + ']'

            self.market_data = self.market_data.loc[self.market_data[self.selection_mode].isin(self.selection), :]
        else:
            self.directory_selected = \
                self.directory_main + '/All'

        Path(self.directory_selected).mkdir(parents=True, exist_ok=True)

    def label_financial_data(self, method='single', delta_to_examine=3, threshold=0.1):
        """This method creates an instance of the class FinancialDataLabeler in order to label market data with a
        specified criterion."""

        if self.directory_selected is None:
            self.apply_selection()

        self.directory_labeled = \
            self.directory_selected + '/method=' + str(method) + ',delta=' + str(delta_to_examine) + \
            ',th=' + str(threshold)

        # Before labeling, see if the file to generate already exists
        if Path(self.directory_labeled + '/labeled_data.csv').is_file():
            self.data = pd.read_csv(self.directory_labeled + '/labeled_data.csv', sep='|')
        else:
            self.market_data = self.market_data.dropna(how='any')
            self.data_labeler = FinancialDataLabeler(self.data, self.market_data, self.deltas, method=method,
                                                     delta_to_examine=delta_to_examine, threshold=threshold)
            self.data_labeler.extract_prices_to_compare()
            self.data_labeler.label_and_join()
            self.data = self.data_labeler.news_data

            # Create a folder for this selection of deltas and these criteria for target labeling
            Path(self.directory_labeled).mkdir(parents=True, exist_ok=True)
            self.data.to_csv(self.directory_labeled + '/labeled_data.csv', sep='|')

        # Print Class Distribution
        print('Class Distribution')
        print(f'    Financial News Labeled with "sell": ' +
              f'{self.data["sell"].sum()} ({(self.data["sell"].sum() * 100 / len(self.data)):.2f}%)')
        print(f'    Financial News Labeled with "buy": ' +
              f'{self.data["buy"].sum()} ({(self.data["buy"].sum() * 100 / len(self.data)):.2f}%)')
        print(f'    Financial News Labeled with "do_nothing": ' +
              f'{self.data["do_nothing"].sum()} ({(self.data["do_nothing"].sum() * 100 / len(self.data)):.2f}%)')
        print(f'{self.data["sell"].sum()} + {self.data["buy"].sum()} + {self.data["do_nothing"].sum()} ' +
              f'= {self.data["sell"].sum() + self.data["buy"].sum() + self.data["do_nothing"].sum()}={len(self.data)}')

    def create_classifier(self, model='bert', max_len=500, validation_size=0.2, batch_size=32, split_type='random',
                          epochs=3):
        """This method uses ktrain and an instance of the class FinancialNewsClassifier create a text classifyer model,
        and acts as an interface to determine its optimum learning rate."""

        self.directory_model = \
            self.directory_labeled + '/model=' + model + ',max_len=' + str(max_len) + ',val_size=' + \
            str(validation_size) + ',batch=' + str(batch_size) + ',split_type=' + split_type + ',epochs=' + str(epochs)

        if Path(self.directory_model + '/predictions.csv').is_file():
            # Recover trained model and predictions
            self.predictions = pd.read_csv(self.directory_model + '/predictions.csv', sep='|')
            self.model = ktrain.load_predictor(self.directory_model + '/predictor')

        elif Path(self.directory_model + '/predictor/config.json').is_file() and Path(
                self.directory_model + '/preprocessing.pkl').is_file():
            # Already trained model but still need predictions
            with open(self.directory_model + '/preprocessing.pkl', 'rb') as f_pickle:
                _, _, preprocessing = pickle.load(f_pickle)

            self.classifier_trainer = FinancialNewsClassifier(self.data, model=model, max_len=max_len,
                                                              validation_size=validation_size, batch_size=batch_size,
                                                              split_type=split_type)
            predictor = ktrain.load_predictor(self.directory_model + '/predictor')
            self.classifier_trainer.get_predictor(self.directory_model, predictor=predictor)
            self.model = self.classifier_trainer.predictor
            self.predictions = self.classifier_trainer.predictions
            self.predictions.to_csv(self.directory_model + '/predictions.csv', sep='|')

        else:
            Path(self.directory_model).mkdir(parents=True, exist_ok=True)
            self.classifier_trainer = FinancialNewsClassifier(self.data, model=model, max_len=max_len,
                                                              validation_size=validation_size, batch_size=batch_size,
                                                              split_type=split_type)

            # If preprocessing has been done before, take advantage of it.
            if Path(self.directory_model + '/preprocessing.pkl').is_file():
                with open(self.directory_model + '/preprocessing.pkl', 'rb') as f_pickle:
                    self.classifier_trainer.train_preprocessed, \
                    self.classifier_trainer.test_preprocessed, \
                    self.classifier_trainer.preprocessing = pickle.load(f_pickle)
            else:
                self.classifier_trainer.preprocess_data()
                with open(self.directory_model + '/preprocessing.pkl', 'wb') as f_pickle:
                    pickle.dump(
                        [self.classifier_trainer.train_preprocessed,
                         self.classifier_trainer.test_preprocessed,
                         self.classifier_trainer.preprocessing], f_pickle)

            self.classifier_trainer.define_model_and_learner()

            # Selection of Learning Rate
            print('Selection of Learning Rate: ')
            max_epochs = input(
                '   If you wish to find it graphically, please enter the maximum number of epochs to iterate '
                'through. Otherwise, press return.')
            try:
                max_epochs = int(max_epochs)
                self.classifier_trainer.request_learning_rate(max_epochs=max_epochs)
                plt.show(block=False)
            except Exception:
                pass

    def train_classifier(self, epochs=10, early_stopping=None):
        """This method trains the classifier, after asking for a valid learning rate from the user."""
        if self.classifier_trainer is not None and self.predictions is None:
            lr = None
            while lr is None:
                lr_input = input('Please enter your desired learning rate:')
                try:
                    lr = float(lr_input)
                except Exception:
                    lr = None
            self.classifier_trainer.train(lr, cycles=epochs, early_stopping=early_stopping,
                                          directory=self.directory_model + '/confusion_matrix.csv')
            plt.show(block=False)

    def predict_with_classifier(self):
        """Given an already trained text classifier, this method invokes it in order to carry out predictions on the
        original dataset."""
        if self.classifier_trainer is not None and self.predictions is None:
            self.classifier_trainer.get_predictor(self.directory_model)
            self.model = self.classifier_trainer.predictor
            self.predictions = self.classifier_trainer.predictions
            self.predictions.to_csv(self.directory_model + '/predictions.csv', sep='|')

    def simulate_portfolio(self, selection=None, starting_amount=100, transaction_amount=1,
                           price='Close', only_validation=False, starting_cash=1e3, start_date=None, end_date=None):
        """This method simulates how a portfolio based on the model's predictions would perform, using an object of
        class PortfolioSimulator."""
        self.directory_portfolio = self.directory_model + '/selection=[' + ','.join([s for s in selection]) + \
            '],starting_stocks=' + str(starting_amount) + ',starting_cash=' + \
            str(starting_cash) + ',transaction=' + str(transaction_amount)

        if not Path(self.directory_portfolio + '/portfolio.csv').is_file():
            self.simulator = PortfolioSimulator(self.data, self.predictions, selection=selection,
                                                starting_amount=starting_amount, transaction_amount=transaction_amount,
                                                price=price, only_validation=only_validation,
                                                starting_cash=starting_cash, start_date=start_date, end_date=end_date)
            self.simulator.insert_prices()
            self.simulator.insert_quantities()
            self.simulator.compute_total()
            self.simulated_portfolio = self.simulator.portfolio

            Path(self.directory_portfolio).mkdir(parents=True, exist_ok=True)
            self.simulated_portfolio.to_csv(self.directory_portfolio + '/portfolio.csv')
        else:
            self.simulator = PortfolioSimulator(self.data, self.predictions, selection=selection,
                                                starting_amount=starting_amount, transaction_amount=transaction_amount,
                                                price=price, only_validation=only_validation,
                                                starting_cash=starting_cash, start_date=start_date, end_date=end_date)
            self.simulator.portfolio = pd.read_csv(self.directory_portfolio + '/portfolio.csv')
            self.simulated_portfolio = self.simulator.portfolio
        self.simulator.visualize()


# df = pd.read_csv('us_equities_news_ultra_short.csv', sep='|', parse_dates=['release_date'])
# f = FinancialNewsPredictor(df)
# f.import_financial_data(deltas=[1, 2, 3, 4, 5, 6, 7, 10, 14])
# f.apply_selection()
# f.label_financial_data(method='MA', delta_to_examine=[6, 7], threshold=0.01)
# f.create_classifier(model='distilbert', max_len=50, batch_size=3, split_type='random')
# f.train_classifier(epochs=1)
# f.predict_with_classifier()
# f.simulate_portfolio(selection='TGT', start_date='2008-11-01', end_date='2009-03-01')
