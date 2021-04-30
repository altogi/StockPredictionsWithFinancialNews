

class FinancialDataLabeler:
    def __init__(self, news_data, market_data, deltas, method='single', delta_to_examine=3, threshold=0.1,
                 base_days=None):
        """
        This class has the task of labelling a dataset of financial news based on whether the prices of a security
        mentioned in the news rise or fall after the date of the news. More specifically, this class will label every
        row in order to indicate whether it is better to sell, buy, or do nothing.
        It does this using two alternative methods:
            1. method='single': The price of the security at delta_to_examine days after each news date is compared with
             the opening price of the new's date.
            2. method='MA': The average of the prices of the delta_to_examine days after each news date is compared
            with the average of a set of days base_days after the news date.
        As a result, for the input:
        - news_data: DataFrame to be labelled, indexed by a column 'id'
        - market_data: DataFrame containing price evolutions based on which to label, indexed by a column 'id'
        - method: 'single' or 'MA'
        - delta_to_examine: This int or list of ints specifies which days after the news should be examined to label.
        All of the values should be included in deltas.
        - threshold: Deviation relative to the base price, at which it is determined that there is a significant
        variation.
        - base_days: If method='NA', this indicates the days from which to compute the reference price of each news. All
         of the values should be included in deltas.
        """
        self.news_data = news_data
        self.market_data = market_data
        self.deltas = deltas
        self.method = method
        self.d_compare = delta_to_examine
        self.th = threshold

        if base_days is None:
            self.d_base = [1, 2, 3, 4, 5, 6, 7]
        else:
            self.d_base = base_days

        # All parameters relative to days after the news date should be set to lists
        if type(self.deltas) == int:
            self.deltas = [self.deltas]
        if type(self.d_compare) == int:
            self.d_compare = [self.d_compare]
        if type(self.d_base) == int:
            self.d_base = [self.d_base]

        # Corresponding columns in self.market_data
        self.cols_compare = ['close_+' + str(d) for d in self.d_compare]
        self.cols_base = None

        # Examine whether all values in self.d_base and self.d_compare are contained in self.deltas
        if any([v not in self.deltas for v in self.d_compare]):
            raise LookupError('Specified value of delta was not found in deltas.')
        if self.d_base is not None:
            if any([v not in self.deltas for v in self.d_base]):
                raise LookupError('Specified value of delta was not found in deltas.')

        # Examine if specified method exists
        if self.method not in ['single', 'MA']:
            raise LookupError('Specified method does not exist.')

        self.indexed_labels = None
        self.market_data.loc[:, 'sell'] = 0
        self.market_data.loc[:, 'buy'] = 0
        self.market_data.loc[:, 'do_nothing'] = 0

    def extract_prices_to_compare(self):
        """This method extracts relevant prices to be compared"""

        if self.method == 'single':
            self.market_data.loc[:, 'price_base'] = self.market_data['open_base'].copy()
        elif self.method == 'MA':
            self.cols_base = ['close_+' + str(d) for d in self.d_base] #+ ['open_base']
            self.market_data.loc[:, 'price_base'] = self.market_data[self.cols_base].mean(axis=1)
        self.market_data.loc[:, 'price_compare'] = self.market_data[self.cols_compare].mean(axis=1)

        self.market_data.loc[:, 'relative_change'] = \
            ((self.market_data['price_compare'] - self.market_data['price_base'])
             / self.market_data['price_base'])

    def label_and_join(self):
        """This method implements the labels and introduces them into the original dataset."""

        self.market_data.loc[self.market_data['relative_change'] < -self.th, 'sell'] = 1
        self.market_data.loc[self.market_data['relative_change'] > self.th, 'buy'] = 1
        self.market_data.loc[
            (self.market_data['relative_change'] <= self.th) & (self.market_data['relative_change'] >= -self.th),
            'do_nothing'
        ] = 1

        self.indexed_labels = self.market_data[['id', 'buy', 'sell', 'do_nothing']].copy()
        self.news_data = self.news_data.merge(self.indexed_labels, on='id')
