import pandas as pd
from datetime import timedelta, datetime
from ipywidgets import interact, SelectionSlider, fixed
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


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
    - start_date: str representing a date, in the form YYYY-mm-dd, at which to start the simulation
    - end_date: str representing a date, in the form YYYY-mm-dd, at which to end the simulation
    """
    def __init__(self, news_data, predictions, selection=None, starting_amount=10, starting_cash=1e6,
                 transaction_amount=1, price='Close', only_validation=False, start_date=None, end_date=None):

        self.transaction = transaction_amount
        self.price = price
        self.starting_amount = starting_amount
        self.starting_cash = starting_cash
        self.content = predictions[['id', 'content']]

        # Extract only news in the validation set
        if only_validation:
            self.predictions = predictions.loc[predictions['is_validation'] == 1, :][['id', 'prediction']]
        else:
            self.predictions = predictions[['id', 'prediction']]

        news_data['release_date'] = pd.to_datetime(news_data['release_date'], utc=False)
        self.data = pd.merge(self.predictions, news_data, left_on='id', right_on='id', how='left') \
            .sort_values(by=['release_date'])[['id', 'prediction', 'ticker', 'release_date']]

        if start_date is not None:
            mask1 = pd.to_datetime(self.data['release_date'], utc=True) >= pd.to_datetime(start_date, utc=True)
            mask1 = mask1.values
        else:
            mask1 = np.array([True for _ in self.data['release_date']])

        if end_date is not None:
            mask2 = pd.to_datetime(self.data['release_date'], utc=True) <= pd.to_datetime(end_date, utc=True)
            mask2 = mask2.values
        else:
            mask2 = np.array([True for _ in self.data['release_date']])

        mask = np.logical_and(mask1, mask2)
        self.data = self.data.loc[mask, :]

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

        self.market_benchmark, self.portfolio_benchmark = None, None

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
            self.portfolio['Price_' + tick].bfill(axis=0, inplace=True)

    def insert_quantities(self):
        """
        This method takes into account the predictions carried out by the model in order to update the quantity of each
        stock that is held, as well as the cash that is left after the transaction. It employs pandas' ffill method in
        order to propagate previous quantities along the self.portfolio DataFrame.
        """
        self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'], utc=True)
        for date, this, operation in zip(self.data['release_date'], self.data['ticker'], self.data['prediction']):
            if this in self.tickers:
                if operation == 'buy':
                    quantity_change = self.transaction
                elif operation == 'sell':
                    quantity_change = -self.transaction
                else:
                    quantity_change = 0
                date = pd.to_datetime(date, utc=True)

                # Insert Operation to Portfolio DataFrame
                self.portfolio.loc[self.portfolio['Date'] == date, 'Operation'] = operation

                # Extract quantity values of all days prior to this news
                to_fill = self.portfolio.loc[self.portfolio['Date'] <= date, self.quantity_cols].copy()

                # Fill unknown quantities with the last existing quantity
                to_fill = to_fill.ffill(axis=0)

                # Update this ticker according to the model's prediction
                if to_fill['Quantity_' + this][len(to_fill) - 1] + quantity_change >= 0:
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

    def visualize(self, interactive=False):
        """
        This method simply serves as a quick visualization of the portfolio performance, comparing it with the returns
        of the S&P 500 as a benchmark. If interactive is True, and if running in an iPython notebook environment,
        a slider for a date selection appears. This slider allows the user to view the news of the dataset closest to
        such date, as well as the predictor's response to the news.
        """

        if self.portfolio.index.name != 'Date':
            self.portfolio['Date'] = pd.to_datetime(self.portfolio['Date'], utc=True)
            self.portfolio.set_index('Date', inplace=True)

        self.market_benchmark = yf.download(
            '^GSPC',
            start=self.data['release_date'].min() - timedelta(days=1),
            end=self.data['release_date'].max()
        )
        self.market_benchmark = (self.market_benchmark[self.price] - self.market_benchmark[self.price][0]) * 100 / \
            self.market_benchmark[self.price][0]

        self.portfolio_benchmark = self.portfolio[self.price_cols].bfill(axis=0).sum(axis=1) * self.starting_amount + \
            self.starting_cash
        self.portfolio_benchmark = (self.portfolio_benchmark - self.portfolio_benchmark[0]) * 100 / \
            self.portfolio_benchmark[0]

        if interactive:
            dates = self.portfolio.index
            options = [(date.strftime(' %d %b %Y '), date) for date in dates]

            selection_range_slider = SelectionSlider(
                value=dates[0],
                options=options,
                description='Select a Date:',
                style={'description_width': 'initial'},
                orientation='horizontal',
                layout={'width': '750px', 'height': '50px'}
            )
            _ = interact(self.return_selected_date, date=selection_range_slider)
        else:
            _, _ = self.plot_single_frame()
            plt.show()

    def plot_single_frame(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.portfolio.index, self.portfolio['Return'], label='Model-Managed Portfolio')
        ax.plot(self.market_benchmark.index, self.market_benchmark, label='S&P 500')
        ax.plot(self.portfolio_benchmark.index, self.portfolio_benchmark, label='Equally Weighted Portfolio')

        if len(self.portfolio) < 10000:
            plt.scatter(self.portfolio.loc[self.portfolio['Operation'] == 'sell', :].index,
                        self.portfolio.loc[self.portfolio['Operation'] == 'sell', 'Return'], label='sell', marker="v",
                        s=80, c='red')
            plt.scatter(self.portfolio.loc[self.portfolio['Operation'] == 'buy', :].index,
                        self.portfolio.loc[self.portfolio['Operation'] == 'buy', 'Return'], label='buy', marker="^",
                        s=80, c='green')

        ax.set_title('Simulated Portfolio Return', fontsize=22)
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Return [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        return fig, ax

    def return_selected_date(self, date):
        date = date.tz_convert('utc')
        all_dates = self.data['release_date'].apply(lambda x: x.tz_localize('utc'))
        closest_index = abs(all_dates - date).argmin()
        closest_date = all_dates.iloc[closest_index]
        content_id = self.data.iloc[closest_index, :]['id']
        ticker = self.data.iloc[closest_index, :]['ticker']
        prediction = self.data.iloc[closest_index, :]['prediction']
        content = self.content.loc[self.content['id'] == content_id, 'content'].values[0]

        fig, ax = self.plot_single_frame()
        ax.plot([closest_date, closest_date],
                [0, self.portfolio['Return'].max()],
                label='selection',
                c='black')
        plt.show()

        print('----------------------------------------------------------------')
        print('COMPANY: ' + ticker + ' - PREDICTION: ' + prediction)
        line_width = 100
        top = 1000
        for cut1, cut2 in zip(range(0, top - line_width, line_width), range(line_width, top, line_width)):
            print(content[cut1:cut2])
