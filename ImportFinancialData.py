import pandas as pd
import yfinance as yf
from datetime import timedelta


class FinancialDataImporter:
    """
    This class is designed to import opening and closing prices of securities specified in a pandas DataFrame
    ticker_dates, at dates specified in the same DataFrame. Moreover, it can also extract prices at certain days delta
    after the specified dates.
    More specifically, for the input:
    - ticker_dates: Pandas DataFrame with three columns. These are a monotonically increasing id ('id'), a string column
    indicating the security's ticker ('ticker'), and a datetime column specifying the reference date of the ticker
    ('release_date)
    - deltas: List of days after the reference date of each ticker for which to obtain the security's price.
    """

    def __init__(self, ticker_dates, deltas=None):
        """Object initialization:
        1. Extract list of unique tickers
        2. Create a DataFrame with the maximum and minimum dates of each ticker
        3. Create columns 'date_base' and 'date_+d', for each of the dates taken into account"""

        self.df = ticker_dates
        self.deltas = deltas

        # List of tickers
        self.tickers = self.df['ticker'].unique()

        # Obtain maximum and minimum dates of news for each ticker. This gives a sense of the time interval
        # for which to look for price data.
        self.df = self.df.rename(columns={'release_date': 'date_base'})
        self.df['date_base'] = pd.to_datetime(self.df['date_base'], utc=True)
        self.df = self.correct_weekends(self.df, date_col='date_base')
        self.min_max_dates = self.df \
            .drop(columns=['id']) \
            .groupby(by=['ticker']) \
            .agg(
                min_date=pd.NamedAgg(column='date_base', aggfunc='min'),
                max_date=pd.NamedAgg(column='date_base', aggfunc='max')
                )

        # Given a specified number of days delta, generate the corresponding dates, posterior to each reference date.
        if self.deltas is not None:
            if type(self.deltas) == int:
                self.deltas = [self.deltas]

            for d in self.deltas:
                col_name = 'date_+' + str(d)
                self.df[col_name] = self.df['date_base'] + timedelta(days=d)
                self.df[col_name] = pd.to_datetime(self.df[col_name], utc=True)
                self.df = self.correct_weekends(self.df, date_col=col_name)

                if d == max(self.deltas):
                    # Update maximum dates based on the maximum number of days after the reference date.
                    self.min_max_dates['max_date'] = self.min_max_dates['max_date'] + timedelta(days=d+2)

        self.price_data = pd.DataFrame(columns=['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    @staticmethod
    def correct_weekends(df, date_col='date'):
        """This simple method intends to correct all dates that fall on a weekend, since no market data will be found
        for them. Saturdays are converted to Fridays, and Sundays to Mondays. As input, the function expects a DataFrame
        with a column, by default called 'date', which has to be corrected when necessary."""

        # Mondays correspond to a 0, Sundays correspond to a 6
        df.loc[:, 'weekday'] = df.loc[:, date_col].dt.weekday.values.copy()

        # Switching Sundays to Mondays
        df.loc[df['weekday'] == 6, date_col] = df.loc[df['weekday'] == 6, date_col].copy() + timedelta(days=1)
        # Switching Saturdays to Fridays
        df.loc[df['weekday'] == 5, date_col] = df.loc[df['weekday'] == 5, date_col].copy() - timedelta(days=1)

        return df.drop(columns=['weekday'])

    def download_prices(self):
        """For each of the tickers included in the original dataset, this function extracts their price data
        for the date intervals contained in self.min_max_dates, using the Yahoo Finance API."""
        for tick in self.tickers:
            data = yf.download(tick,
                               start=self.min_max_dates.loc[tick, 'min_date'],
                               end=self.min_max_dates.loc[tick, 'max_date'])
            data['Ticker'] = tick
            tick_obj = yf.Ticker(tick).info
            if 'sector' in tick_obj.keys():
                data['Sector'] = tick_obj['sector']
                data['Industry'] = tick_obj['industry']
            else:
                data['Sector'] = 'Unknown'
                data['Industry'] = 'Unknown'
            data.reset_index(inplace=True)
            self.price_data = pd.concat([self.price_data, data], ignore_index=True)
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'], utc=True)
        print('Finalized download of market data.')

    def insert_prices(self):
        """This function inserts the previously generated price data into the original dataset, making sure that
         both ticker and dates match. First it performs an inner join to obtain the closing and opening
         prices of the reference date. Then, for each of the considered values of delta to take into account,
         it carries out a left join of the original dataset with the price data to introduce the Adj Close data of the
         date delta days after the base date."""

        columns_to_keep = list(self.df.columns) + ['Sector', 'Industry', 'Close']

        # Join for reference date, obtaining Adj Close and Open
        self.df = pd.merge(self.df,
                           self.price_data,
                           how='inner',
                           left_on=['ticker', 'date_base'],
                           right_on=['Ticker', 'Date']
                           )[columns_to_keep + ['Open']]
        self.df.rename(columns={
                                'Close': 'close_base',
                                'Open': 'open_base',
                                'Sector': 'sector',
                                'Industry': 'industry'
                                }, inplace=True)

        # Join for each of the delta values
        for d in self.deltas:
            columns_to_keep = list(self.df.columns) + ['Close']
            col_name = 'date_+' + str(d)
            self.df = pd.merge(self.df,
                               self.price_data,
                               how='left',
                               left_on=['ticker', col_name],
                               right_on=['Ticker', 'Date'])[columns_to_keep]
            self.df.rename(columns={'Close': 'close_+' + str(d)}, inplace=True)
