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
    <a href="https://github.com/altogi/StockPredictionsWithFinancialNews/blob/main/Prediction of Stock Market Evolutions with Financial News.ipynb">View Demo</a>
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
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
* `validation_size`: This float sets the size of the validation set used to evaluate the model. 
* `batch_size`: This int determines how many documents are bundled together at each iteration of training.




Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_





<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/altogi/
[product-screenshot]: images/screenshot.png
