import pandas as pd
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt


class PortfolioSimulator:
    """
    This class is used to find out how well a portfolio that follows the predictions made by a text classifier
    from class FinancialNewsClassifier works out. This simulation works simply by buying or selling according to the
    model's signals, and based on these orders, update the portfolio's equity and cash taking into account the market
    price of every equity. To simplify, no transaction costs nor inflation are taken into account, although a realistic
    approach would require so.
    With regards to the class' input arguments:
    - news_data: This pandas DataFrame contains a set of financial news, identified by a column 'id', organized
    with a column 'release_date' including the news' date, and assigned to a particular security with a column 'ticker'.
    - predictions: This pandas DataFrame contains predictions of the text classification model obtained from
    FinancialNewsClassifier on the previous dataframe. It includes an identifying column 'id', a prediction column
    'prediction' with a prediction string ('buy', 'sell', or 'do_nothing') and a column 'is_validation' indicating
    whether the particular row has not been used for the training of the model.
    - selection: This list of strings allows the user to select only a set of tickers to simulate their performance. If
    None, all companies are taken into account.
    - starting_amount: int, Indicates the number of stocks of each security that the simulation begins with
    - starting_cash: float, Indicates the starting cash amount of the simulation
    - transaction_amount: int, Indicates how many stocks are involved in each transaction
    - price: string, Indicates what stock price should be taken for a daily reference. To choose between 'Close',
    'Open', and 'Adj Close'.
    - only_validation: boolean, If True, the simulation will only be done with financial news belonging to the
    validation set of the model obtained from FinancialNewsClassifier
    """
    def __init__(self, news_data, predictions, selection=None, starting_amount=10, starting_cash=1e6,
                 transaction_amount=1, price='Close', only_validation=False):

        self.transaction = transaction_amount
        self.price = price

        # Extract only news in the validation set
        if only_validation:
            self.predictions = predictions.loc[predictions['is_validation'] == 1, :][['id', 'prediction']]
        else:
            self.predictions = predictions[['id', 'prediction']]

        news_data['release_date'] = pd.to_datetime(news_data['release_date'], utc=False)
        self.data = pd.merge(self.predictions, news_data, left_on='id', right_on='id', how='inner') \
            .sort_values(by=['release_date'])[['id', 'prediction', 'ticker', 'release_date']]

        if selection is None:
            self.tickers = self.data['ticker'].unique().tolist()
        else:
            if type(selection) == str:
                self.tickers = [selection]
            elif type(selection) == list:
                self.tickers = selection

        # Store column names for later
        self.quantity_cols = ['Quantity_' + ticker for ticker in self.tickers]
        self.price_cols = ['Price_' + ticker for ticker in self.tickers]

        # Initialize portfolio dataframe
        self.portfolio = pd.DataFrame(columns=['Date', 'Cash', 'Operation'] + self.quantity_cols)
        self.portfolio['Date'] = pd.date_range(start=self.data['release_date'].min() - timedelta(days=1),
                                               end=self.data['release_date'].max())
        self.portfolio.iloc[0, 3:] = starting_amount
        self.portfolio.iloc[0, 1] = starting_cash
        self.portfolio.iloc[0, 2] = 'do_nothing'

    def insert_prices(self):
        """
        This method utilizes the Yahoo Finance API in order to download the prices of the securities involved in the
        simulation, for the date range of self.data. These prices are introduced in self.portfolio for further
        calculations.
        """
        for tick in self.tickers:
            data = yf.download(tick,
                               start=self.data['release_date'].min() - timedelta(days=1),
                               end=self.data['release_date'].max()
                               ) \
                .reset_index() \
                .rename(columns={self.price: 'Price_' + tick})[['Date', 'Price_' + tick]]
            self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'], utc=True)
            data['Date'] = pd.to_datetime(data['Date'], utc=True)
            self.portfolio = pd.merge(self.portfolio, data, on='Date', how='left')
            self.portfolio['Price_' + tick].ffill(axis=0, inplace=True)

    def insert_quantities(self):
        """
        This method takes into account the predictions carried out by the model in order to update the quantity of each
        stock that is held, as well as the cash that is left after the transaction. It employs pandas' ffill method in
        order to propagate previous quantities along the self.portfolio DataFrame.
        """
        for date, this, operation in zip(self.data['release_date'], self.data['ticker'], self.data['prediction']):
            if this in self.tickers:
                if operation == 'buy':
                    quantity_change = self.transaction
                elif operation == 'sell':
                    quantity_change = -self.transaction
                else:
                    quantity_change = 0

                # Insert Operation to Portfolio DataFrame
                self.portfolio.loc[self.portfolio['Date'] == date, 'Operation'] = operation

                # Extract quantity values of all days prior to this news
                to_fill = self.portfolio.loc[self.portfolio['Date'] <= date, self.quantity_cols].copy()

                # Fill unknown quantities with the last existing quantity
                to_fill = to_fill.ffill(axis=0)

                # Update this ticker according to the model's prediction
                to_fill['Quantity_' + this][len(to_fill) - 1] = \
                    to_fill['Quantity_' + this][len(to_fill) - 1] + quantity_change

                # Insert this into the portfolio dataframe
                self.portfolio.loc[self.portfolio['Date'] <= date, self.quantity_cols] = to_fill.copy()

                # Similarly, update the corresponding Cash column with the transaction's money
                cashflow = self.transaction * self.portfolio.loc[self.portfolio['Date'] == date, 'Price_' + this].values
                to_fill = self.portfolio.loc[self.portfolio['Date'] <= date, 'Cash'].copy().ffill(axis=0)
                to_fill[len(to_fill) - 1] = to_fill[len(to_fill) - 1] + cashflow

                self.portfolio.loc[self.portfolio['Date'] <= date, 'Cash'] = to_fill.copy()

    def compute_total(self):
        """
        Taking into account the evolution of security prices and stocks held, this method computes the evolution of
        portfolio value by multiplying each column representing the quantity of stock with its corresponding price
        column, and summing along all rows. It also takes into account the evolution of held cash, in order to find
        the percentage change of the portfolio value.
        """
        # Prepare portfolio dataframe for aggregation
        self.portfolio.set_index('Date', inplace=True)

        # Multiply all quantity columns by all price columns, to obtain the portfolio's total worth
        self.portfolio['Total_Invested'] = self.portfolio[self.quantity_cols] \
            .rename(columns={'Quantity_' + tick: tick for tick in self.tickers}) \
            .mul(
            self.portfolio[self.price_cols]
                .rename(columns={'Price_' + tick: tick for tick in self.tickers}),
            axis=0, fill_value=0
        ).sum(axis=1)

        self.portfolio['Total'] = self.portfolio['Total_Invested'] + self.portfolio['Cash']
        self.portfolio['Return'] = (self.portfolio['Total'] - self.portfolio['Total'][0]) * 100 \
                                   / self.portfolio['Total'][0]

    def visualize(self):
        """
        This method simply serves as a quick visualization of the portfolio performance, comparing it with the returns
        of the S&P 500 as a benchmark.
        """
        benchmark = yf.download(
            '^GSPC',
            start=self.portfolio.index.min() - timedelta(days=1),
            end=self.portfolio.index.max()
        )

        benchmark = (benchmark[self.price] - benchmark[self.price][0]) * 100 / benchmark[self.price][0]

        plt.figure(figsize=(15, 10))

        plt.plot(self.portfolio.index, self.portfolio['Return'], label='Portfolio Return')
        plt.plot(benchmark.index, benchmark, label='S&P 500')
        plt.scatter(self.portfolio.loc[self.portfolio['Operation'] == 'buy', :].index,
                    self.portfolio.loc[self.portfolio['Operation'] == 'buy', 'Return'], label='buy', marker="^",
                    s=80, c='green')
        plt.scatter(self.portfolio.loc[self.portfolio['Operation'] == 'sell', :].index,
                    self.portfolio.loc[self.portfolio['Operation'] == 'sell', 'Return'], label='sell', marker="v",
                    s=80, c='red')

        plt.title('Simulated Portfolio Return', fontsize=22)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Return [%]', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.show()

