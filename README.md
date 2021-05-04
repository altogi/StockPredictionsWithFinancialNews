<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]

<p align="center">

  <h1 align="center">Predicting Stock Market Trends with Financial News</h3>

  <p align="center">
    An application of BERT to profitable trading.
    <br />
    <a href="https://github.com/altogi/StockPredictionsWithFinancialNews/blob/main/Prediction_of_Stock_Market_Evolutions_with_Financial_News.ipynb">View Demo</a>
    ·
    <a href="https://github.com/altogi/StockPredictionsWithFinancialNews/issuess">Report Bug</a>
    ·
    <a href="https://github.com/altogi/StockPredictionsWithFinancialNews/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#1-selecting-a-dataset">1. Selecting a Dataset</a></li>
        <li><a href="#2-creating-a-financialnewspredictor-object">2. Creating a `FinancialNewsPredictor` Object</a></li>
        <li><a href="#3-importing-financial-data">3. Importing Financial Data</a></li>
        <li><a href="#4-labeling-articles-according-to-price-data">4. Labeling Articles According to Price Data</a></li>
        <li><a href="#5-defining-the-text-classifier">5. Defining the Text Classifier</a></li>
        <li><a href="#6-training-and-predicting">6. Training and Predicting</a></li>
        <li><a href="#7-simulating-a-model-managed-portfolio">7. Simulating a Model-Managed Portfolio</a></li>
      </ul>
    </li>
    <li><a href="##project-structure">Project Structure</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

In the stock market, information is money. Receiving the information first gives one a significant advantage over other traders.
Thus, it makes sense that financial news have a great influence over the market.

Given the recent rise in the availability of data, and the apparition of revolutionary NLP techniques, it has been attempted in many occasions to predict market trends based on financial news. The majority of the existing solutions rely on sentiment analysis, assuming that a positive document sentiment is directly related to increases in a security's price, and viceversa. Sentiment is either extracted using a predefined dictionary of tagged words, or by applying deep learning techniques that rely on a large datasets of labeled news. An advanced example of rule-based sentiment analysis is [VADER](https://blog.quantinsti.com/vader-sentiment/), a model that is sensitive not only to polarity, but also to a document's intensity.

With the recent dawn of the Transformer, it is now possible to extract the sentiment from a document in a much quicker non-sequential procedure, and with the usage of pre-trained models such as BERT, applying these models to a desired use case has never been simpler. An example of this is [FinBERT](https://arxiv.org/abs/1908.10063), a text classifier predicting sentiment with a fine-tuned version of BERT.

Nevertheless, sentiment can act as an intermediate factor between the news, and the stock's price. As a result, developing a text classifier to predict sentiment is not as efficient as directly predicting price evolutions, when the objective is to develop a profitable trading strategy.

This work has implemented a text classifier based on BERT, fine-tuned with a dataset of financial news, and trained in order to predict whether a stock's price will rise or fall. As opposed to FinBERT, sentiment is not taken into account. Instead, a set of criteria based on the price evolutions close to the release date of every news article have been applied, in order to label the dataset with which BERT is fine-tuned. Moreover, this work has been developed based on a much more extensive dataset than the one used for FinBERT, thus further capturing the uncertainties of the market. In consequence, it is possible to profitably manage a portfolio relying exclusively on this text classifier, without complementing it with other trading strategies, as many sentiment-based trading applications do.


<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple example steps.
### Prerequisites
These are some things that you will need before locally setting up this application:

* **Memory Requirements:** This project was completely run and tested on Google Colab, making use of its 12GB NVIDIA Tesla K80 GPU. Although the application can be run with less memory-consuming models that do not require a GPU (DistilBERT), it is recommendable to use similar levels of RAM, especially for large datasets.
* **yfinance:** Install with the following command.
  ```sh
  pip3 install yfinance
  ```
* **TensorFlow 2:** If not already installed.
  ```sh
  pip3 install tensorflow
  ```
* **ktrain:** Install with the following command.
  ```sh
  pip3 install ktrain
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/altogi/StockPredictionsWithFinancialNews.git
   ```
3. Install yfinance and ktrain.
4. Enjoy!

<!-- USAGE EXAMPLES -->
## Usage
### 1. Selecting a Dataset
To use this application, a dataset of financial news is needed. This is the dataset on which the text classification model will be trained, and later on validated. This dataset has to have the following features:
* An identification column **id**, representing every news article.
* A column **content** containing the article's text.
* A field **ticker** with the stock ticker of the company mentioned in the article.
* A column **release_date** with the date in which the article came out.

A great dataset to use for this application is *us_equities_news_dataset.csv* from Kaggle's [Historical financial news archive](https://www.kaggle.com/gennadiyr/us-equities-news-data). This dataset is a news archive of more than 800 american companies for the last 12 years, and has been used in every step of the development of this project.

### 2. Creating a `FinancialNewsPredictor` Object
This object will carry out all of the steps of the application, and thus its correct definition is very important. Besides taking the aforementioned dataset as input, these three parameters are required to ensure the execution goes as desired:
1. `base_directory`: This is the root directory in which all of the files generated during the application's execution will be stored. By default, this is set to be the directory in which the application is located.
2. `selection`: To tailor the model to a reduced number of companies, it is possible to apply a selection of tickers instead of all of the companies included in the financial news dataset. As a list of strings, one can specify a number of tickers, a number of sectors, or a number of industries, in order to filter the dataset according to such selection.
3. `selection_mode`: To specify the selection mode, one must enter either 'ticker', 'industry', or 'sector' for this parameter, thus letting the application know what the selection stands for.

The code snippet below shows how one would create a `FinancialNewsPredictor` object focusing on companies from the technology or financial services sectors.

```
f = FinancialNewsPredictor(df_news, 
                         base_directory='./BERTforMarketPredictions',
                         selection=['Technology', 'Financial Services'],
                         selection_mode='sector')
```

### 3. Importing Financial Data

Now it is necessary to define an instance of class `FinancialDataImporter` through a method `import_financial_data()` of `FinancialNewsPredictor`, in order to import price data based on the dates and stock tickers associated to each news article. To do so, this class simply uses the Yahoo Finance API. 

The only parameter for method `import_financial_data()` is `deltas`. This list of integers specifies what market days after each new's release date are considered relevant. The closing market price of each of these relevant dates is extracted for further analysis in the application.

This method also makes sure to store the resulting market data in a table *market_data.csv*, which itself is also in a folder whose name is determined by the selected `deltas`. This way, if the same `deltas` are used in a different occasion, the dataset can be taken advantage of, instead of performing the same computations all over again.

For example, with `deltas = [1, 2]`, one would execute the following line of code:

```
f.import_financial_data(deltas)
```

And taking into account the `base_directory` specified earlier, there would be a table *market_data.csv* in *./BERTforMarketPredictions/deltas=1,2/*, with columns:
* id
* ticker
* sector
* industry
* date_base (associated article's release date)
* date_+1
* date_+2
* open_base (opening price at the associated article's release date)
* close_base (closing price at the associated article's release date)
* close_+1 (stock's closing price a day after the associated article's release date)
* close_+2 (stock's closing price two days after the associated article's release date)

### 4. Labeling Articles According to Price Data

Based on the recently imported prices near the date of each article, it is necessary to apply a set of criteria to evaluate whether, given a price evolution after the article's release date, one should buy or sell stock of the mentioned company based on such article. More precisely, this step of this recipe takes as input a set of prices at the specified `deltas` and labels each news article with "buy", "sell" or "do_nothing". This labeling is exactly the target variable with which the text classifier will train later on. Method `label_financial_data()` of `FinancialNewsPredictor` carries out this step.

At its core, this step simply involves the comparision of two prices: a price representative of the company before the news came out, and a price representing the market's reaction to the news. If the latter is above a certain threshold from the former, it is adviseable to label the news with "buy". Alternatively, if the stock's price dips below a certain threshold after the news, the news will be labeled with "sell". Otherwise, "do_nothing" will be applied. This threshold is specified as a relative variation between the latest price with respect to the earliest via `threshold`.

According to the user's input parameters, two different labeling schemes can be followed:
* Simple Method: If `method ='single'`, the opening price of the new's release date is compared with the closing price of another date. This second date to compare with is specified with `delta_to_examine`. For example, if `delta_to_examine = 3`, the opening price of the new's release date will be compared with the closing price three days after the release date (as long as 3 is included in the employed `deltas`).
* Moving-Averages Method: If `method = MA`, two averages of prices are compared. Through `base_days`, the closing prices of a set of days within `deltas` is averaged to obtain a price representing the market before the news came out. Also, through `delta_to_examine`, the closing prices of a set of days within `deltas` is averaged to obtain a price representing the market after the news came out. These two average prices will be compared to label the news in a manner that is less sensitive to outlier prices. For instance, if `base_days = [1, 2, 3, 4, 5, 6, 7]` and `delta_to_examine = [6, 7]`, the average closing prices at the end of the week after the new's release date will be compared with the average closing prices of the whole week. 

As an example, if one were to label based on the simpler model, comparing the new's opening price with the closing price 2 days after the release date, one would use the following line of code:
```
f.label_financial_data(method='single', delta_to_examine=2, threshold=0.1)
```
As a result, following the previous examples, a new table *labeled_data.csv* will be stored in *./BERTforMarketPredictions/deltas=1,2/method=single,delta=3,th=0.1*, with the labeled dataset. This table will be re-used in other occassions in which `deltas` and the labeling method remain constant.

### 5. Defining the Text Classifier

Before a text classifier model can be trained, method `create_classifier()` of `FinancialNewsPredictor` is necessary. These are the main goals of this method:
1. To ensure that all required user input is properly specified.
2. To carry out the preprocessing of the dataset according to the selected model.
3. To define a ktrain model and a learner to train.
4. To aid in the selection of a suitable learning rate for the training to come.

The input parameters that this step receives greatly influence how the text classifier model is trained. These are:
* `model`: This string indicates what model should the text classifier be based on. According to ktrain's source code, the available models are: 'fasttext' for a FastText model, 'nbsvm' for a NBSVM model, 'logreg' for logistic regression using embedding layers, 'bigru' for Bidirectional GRU with pretrained word vectors, 'bert' for BERT Text Classification and 'distilbert' for Hugging Face's DistilBert model.
* `max_len`: This int indicates the maximum number of words that can be taken into account per document when training the model.
* `validation_size`: This float sets the size of the validation set used to evaluate the model. By default it is 0.2.
* `batch_size`: This int determines how many documents are bundled together at each iteration of training.
* `split_type`: Specifies how to split the data into training and validation sets. `split_type='random'` simply
randomly splits the input dataset according to the specified `validation_size`, ensuring an homogeneous distribution of each class. `split_type='time_series'` ensures that only the latest entries of the dataset are taken as a validation set. By default it is 'random'.

For a discussion regarding the constraints of your sytem's RAM on the possible combinations of `max_len` and `batch_size` when `model = 'bert'`, [click here](https://github.com/google-research/bert#out-of-memory-issues).

With regards to the preprocessing of the dataset, the method `text.texts_from_df()` from ktrain is used, and it is made sure that if previous preprocessings have been carried out for the same model and combination of parameters, it can be reloaded instead of recomputed. The preprocessing files will be stored in a folder specific to the current combination of `model`, `max_len`, `validation_size`, `batch_size`, and `split_type`. Similarly, in the case in which the model has already been trained and its corresponding predictor has been stored, it is loaded alongside its predictions (if already computed), and thus the definition of a ktrain model and learner for further training is avoided.

In case the training of a new model is necessary, after defining the required ktrain learner and model, this method goes on to aid the user in specifying a learning rate. At this point, the user has the option of allowing the application to iterate throughout several values of this parameter to obtain the loss-learning rate curve. Based on this curve, the user is expected to estimate an adequate value for this parameter. [Generally, the maximum learning rate associated with a decreasing loss with increasing learning rate is most adequate.](https://arxiv.org/pdf/1506.01186.pdf)

Following the previous series of examples, to define a text classifier based on BERT, with a maximum sequence length of 256 and a batch size of 16, one would run the following line of code:
```
f.create_classifier(model='bert', max_len=256, batch_size=16)
```

### 6. Training and Predicting

Once an appropriate learning rate has been determined, the learner can be trained with method `train_classifier()` of `FinancialNewsPredictor`. Here, the user has to specify the maximum number of epochs `epochs` to train as well as the number of epochs `early_stopping` after which to stop if no improvement in the validation loss has occurred. Before training, the user will be prompted to input the desired learning rate.

Training is carried out with ktrain's `autofit()` method, which implements a triangular learning rate policy. In other words, every epoch is divided into two halves: in the first half, the learning rate increases linearly from a base rate to the learning rate specified by the user, whereas in the second half it decreases linearly from such maximum to a near-zero rate. This training scheme was chosen since [it is well suited for Keras built-in training callbacks, such as `EarlyStopping`.](https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c?gi=ea843ab1ae3c). This learning rate policy was proposed by Leslie Smith of the Naval Research Laboratory, and her work can be found [here](https://arxiv.org/pdf/1506.01186.pdf).

Once the application stops training, (either because of a converged model or because of an insufficient numer of epochs), the training curve of the training is presented, and a confusion matrix is stored in *confusion_matrix.csv*, in the same folder as the preprocessing files.

For instance, to train during 6 epochs with a patience of 2 epochs, run the following line of code:

```
f.train_classifier(epochs=6, early_stopping=2)
```

Once the model has been trained, predicting simply involves applying the model to the whole labeled dataset. As a result, the dataset of financial news will be labeled not only according to the price criteria specified earlier on, but also by the model's predictions. This resulting dataset includes a column 'is_validation' that indicates whether each news article was used for training the model or not. These predictions are stored in a table *predictions.csv* under the same folder as all model files. The ktrain predictor resulting from training is also stored under the same directory, in a folder *predictor*. To predict, simply run the following:

```
f.predict_with_classifier()
```

### 7. Simulating a Model-Managed Portfolio

To validate the obtained predictions, it makes sense to simulate the real-life performance of a portfolio that is exclusively managed by the model's predictions. Since in the end the accuracy of the model is also significantly dependent on the price criteria applied to label the dataset, simulating the resulting portfolio is a way to validate both the trained model and the price criteria themselves at the same time. To this end, a method `simulate_portfolio` of `FinancialNewsPredictor` can be used.

The simulation starts with a portfolio made out of an equal number of a specified set of stocks, as well as a specified amount of cash. For any point in the simulation, the portfolio's value is the sum of the market value of every stock in the portfolio and the current amount of cash owned. Financial news and their predictions are processed in the ascending order of their release dates. In this way, for each of the predictions related to the stocks in the simulated portfolio, three actions are possible:
* If the prediction is "buy", the quantity of the stock mentioned in the news is increased, and the cash amount is reduced according to the stock's market price.
* If the prediction is "sell", the quantity of the stock mentioned in the news is decreased, and the cash amount is increased according to the stock's market price.
* If the prediction is "do_nothing", nothing is changed.

In case the quantity of a particular stock in the portfolio is zero, no sale will take place until the quantity is increased. In case the portfolio's cash amount is zero, no stocks will be bought until this amount is increased. This dynamic goes on until the last of the model's predictions has been processed. Once the simulation is finalized, the portfolio return is compared with the starting portfolio left untouched and a benchmark index (S&P 500). Moreover, a table *portfolio.csv* with the portfolio's evolution during the simulation is stored in a folder specific to the parameters of the simulation.

Based on this simulation dynamic, the method's input arguments are the following:
* `selection`: This list of strings allows the user to select only a set of tickers to simulate their performance. If None, all companies are taken into account.
* `starting_amount`: int, Indicates the number of stocks of each company that the simulation begins with.
* `starting_cash`: float, Indicates the starting cash amount of the simulation.
* `transaction_amount`: int, Indicates how many stocks are involved in each transaction.
* `price`: string, Indicates what stock price should be taken for a daily reference. To choose between 'Close', 'Open', and 'Adj Close'.
* `only_validation`: boolean, If True, the simulation will only be done with financial news belonging to the validation set of the model obtained from `FinancialNewsClassifier`.
* `start_date`: str representing a date, in the form YYYY-mm-dd, at which to start the simulation.
* `end_date`: str representing a date, in the form YYYY-mm-dd, at which to end the simulation.

For example, in order to simulate a portfolio made up of Apple, Microsoft, Amazon, and Tesla stocks, ranging from 2015 to 2020, starting with 1000 stocks of each, one has simply to run:

```
f.simulate_portfolio(starting_amount=1000, 
                     start_date='2015-01-01', 
                     end_date='2020-01-01', 
                     selection=['AAPL', 'MSFT', 'AMZN', 'TSLA']
                     )
```


## Project Structure


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/altogi/
