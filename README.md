# 4641-Group-Project
## Team Members: JayaSai Somasundaram, Roman Karmazin, Anna Nguyen, Wontaek Kim

GitHub Pages link: https://github.gatech.edu/pages/anguyen373/4641-Group-Project/ (open to all enterprise members)

# Infographic
![Predicting Stock Based on impactful features](https://github.gatech.edu/storage/user/59946/files/9d7b8403-604a-4ddb-b84b-05babe1eedfb)

# Introduction

Machine learning-based stock price prediction helps us determine the future worth of a company's shares and other financial assets. However, the whole point of stock market forecasting is making a lot of money. Therefore, our project aims to develop a modern price trend prediction model focusing on short-term price trend projection.

The objective of this project is to use unsupervised and supervised learning methods to compare the trends of FAANG stocks. We use linear (PCA) and nonlinear (t-SNE) dimensionality reduction techniques for unsupervised learning. While in recent years, principal component analysis (PCA) has been widely applied to the study of financial markets. However,  the capability of the t-SNE algorithm can show a better visualization and a more significant improvement in clustering quality compared with Principal Component Analysis (PCA). Since our focus is on a series of the most important features through dimension reduction, and by using these features, we have clustered them to find the relationships between the stocks and their trends. Initially, the primary focus was on the K-Means algorithm, but we also used Gaussian Mixture Model to compare the results. For supervising, we utilize  Linear Regression, and Neural Networks, and the Long Short-Term Memory (LSTM) technique. 



# Methods
The dataset that we are using:

Kaggle - FAANG (FB,Amazon,Apple,Netflix,Google) Stocks 


https://www.kaggle.com/datasets/kaushiksuresh147/faang-fbamazonapplenetflixgoogle-stocks?resource=download
 

Our project is complicated as our objective is price direction predictions utilizing both unsupervised and supervised machine learning. Unsupervised machine learning is used with dimensionality reduction to pre-process the dataset accurately and find the most essential features. It also clusters price points and possibly shows what prices a stock will likely go up or down. The supervised machine learning utilizes regression to predict the next data point based on a given time increment. There are visualizations for the raw data, dimensionality reduction, and algorithmic outputs for each of the five FAANG stocks.



## Data Collection and Preprocessing: 

The first step was how we made our dataset. We are dealing with time series data when working with stock prices. When dealing with time series, the first challenge is formulating the problem. Because time series data is difficult to represent, we hold our data as discrete values. 


## Time Series to Discrete:

The key is to convert a dataset to discrete data from time series using fixed value intervals. Thus, we selected our dataset using discrete data points on a daily time frame. Once we realized this, we shifted away from time-series data representations and found a dataset that showed discrete data points daily for FAANG stock names. We then stored each FAANG name as a separate CSV file for the given period we chose for our data.

## Original Dataset:
Initially, we had five datasets for each of the five stocks. As mentioned above, these datasets had discrete data points on a daily interval. The features of this dataset are known in the stock world as OHLC. An OHLC chart is a bar chart that shows open, high, low, and closing prices for each period, with the closing price being considered the most important by many traders. Overall, it helps interpret the day-to-day sentiment of the market and forecast future price changes through the patterns produced. 


## Unsupervised
For unsupervised, we are going to compare the trends of each stock on Facebook, Amazon, Apple, Netflix, and Google stocks (FAANG) by year and analyze how various features, such as properties of each company, the occurrence of the product launch of each company and new events, affect the price of each stock. Further, we can cluster the different stocks based on each feature to investigate how they are similar in certain features. The stock price varies depending on diverse elements; therefore, we will look at those features in further detail. 

&emsp; &emsp; Unsupervised Learning: K-means:

We are using a KMeans hard clustering algorithm first. The algorithm will be trained till the loss function stops decreasing as iterations increase. The cap on iterations, however, will be at 10000 due to computing capacity constraints and time constraints. Therefore, the results will be verified in two different ways. First, we will see what the loss function brings down to so then we can evaluate the algorithm's effectiveness and how it will be trained off the given data. Additionally, the dataset will be labeled based on the direction of different aggregate timeframes and plotted like that. Then the output of KMeans will be compared to that plot to see if there is any resemblance. The data points will then be marked as in the right cluster or wrong cluster, and an aggregate measure of accuracy will be taken.

&emsp; &emsp; Unsupervised Learning: GMM:

The dimensionality reduction used on the GMM clustering was the same as the K-means one. In addition, the dataset was reduced down to two features to allow two-dimensional visualizations. Also, the data labels are based on change from one day to the next, then classifying above 1% as up, below 1% as down, and below these values as neutral. Finally, we allowed the model to train until the loss value flat lines, and the algorithm could no longer get any more accurate. Then, we set the max iterations to a very high value to allow the loss of a flat line. Two-dimensional visualization of the reduced features outputted from PCA allowed us to compare accurately with the KMeans algorithm. 

&emsp; &emsp; Unsupervised Learning: PCA:

Principal Component Analysis (PCA) is used to identify a smaller number of uncorrelated variables known as principal components from a more extensive data set. PCA is a standard technique for visualizing high-dimensional data and for data pre-processing. PCA reduces the dimensionality of a data set by maintaining as much variance as possible. Our visualization is a 2D plot of feature one vs. feature two, enabling us to obtain two-dimensional clusters.

&emsp; &emsp; Unsupervised Learning: t-SNE:

(t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm for exploring high-dimensional data. This algorithm finds patterns in the data by identifying observed clusters based on the similarity of data points with multiple features. It also maps the multi-dimensional data to a lower dimensional space, so the input features are no longer identifiable. Thus we cannot make any inference based only on the output of t-SNE. So essentially, it is mainly an excellent data exploration and visualization technique.



## Supervised

For supervised, stock market prediction aims to anticipate the future movement of a financial exchange's stock value. Investors will be able to make more money if they can accurately forecast share price movement. For example, we are going to predict the price of the FAANG stocks when some features are changed or added. We may get insight into market behavior over time with an adequate stock prediction model, recognizing tendencies that might otherwise go unnoticed. With the computer's increased processing capability, machine learning will be an effective way to overcome this challenge.

For the supervised learning algorithm, we will use Linear Regression and Neural Networks. 

# Results 

The first diagram for each stock shows a plot of raw data tagged based on the day-to-change percent change. The green color is for increased more than 1%, red for decreased more than 1%, and yellow for between -1% and +1%.

Then the data were clustered using GMM and KMeans. With GMM, the data would not cluster if it wasn’t normalized due to the variance in the range of the two feature outputs from PCA. With KMeans, both normalized and unnormalized clustered, but the normalized model was much more similar to that of GMM. An observation in the data was that for all models, the loss value of the GMM was relatively lower than the loss value of the KMeans.

Based on the visual analysis, a few other things can be seen:

&emsp; &emsp; 1.	The raw data seems to have no pattern based on how it was labeled, which could be the issue.

&emsp; &emsp; 2.	The non-normalized KMeans output seems to be drawing a vertical line right along the middle of the dataset, which should not be happening.

&emsp; &emsp; 3.	The clusters it is deducing are also highly skewed due to the lack of normalization affecting Euclidean distance.

When looking at the output of the normalized models, it seems that KMeans is still drawing lines within the data and setting almost a rigid boundary. However, GMM, on the other side, seems to be clustering quite accurately when visually observing the dataset.

&emsp; &emsp; Results for KMeans and GMM

![image](https://github.gatech.edu/storage/user/59946/files/88532a7b-3c29-404f-869e-41db709b0cf7)

&emsp; &emsp; Results for PCA and t-SNE

![image](https://github.gatech.edu/storage/user/59946/files/5ddeae04-8c32-4874-8054-02f61c876cb0)

We constructed a Histogram, which groups the data into bins and is the fastest way to get an idea about the distribution of each attribute in the dataset. Besides, histograms also help us to see possible outliers.

![image](https://github.gatech.edu/storage/user/59946/files/ca90aa69-736d-4fb2-9366-f6ba14b6a9a7)

For supervised learning, we used LSTM and a basic regression model on our dataset. Based on the visualizations generated, our models demonstrated significant overfitting. 

![image](https://github.gatech.edu/storage/user/59946/files/b49d120b-af41-444c-9c68-f1a35595c048)


# Discussion

&emsp; &emsp; Feature Selection

We selected the most relevant and important features to our model, including volatility, open, close, high, and low prices. We dropped dividends, market cap, and shares because of the absence of data related to dividends and shares. After reducing the dimensions down to two features, we generated visualizations to observe the relationships between the features. Unfortunately, we could not establish a positive relationship between the features due to the abstract clustering we produced.We used both t-SNE and PCA dimensionality reduction algorithms to compare and observe the differences. Based on the visualizations below, both algorithms clustered fairly well while simultaneously adhering
to their specific properties. 
 

&emsp; &emsp; Unsupervised Learning
	
For unsupervised learning, we used clustering, specifically the GMM and K-means algorithms. We mainly evaluated the performance of our clustering through visualizations. The more separated and visibly clustered our data is, the better the clustering algorithm performed. Based on the visualizations generated, K-means clustered better than GMM, which should be expected because K-means clusters are based on hard clustering, whereas GMM clusters are based on probability and soft clustering. It is difficult to evaluate the performance of our clustering based on visualizations. We clustered our data after reducing the dimension to two features, but clustering it does not seem to provide any actual meaning. However, we can make educated guesses based on our clusters and data we fed the model and hypothesize that volatility significantly impacts the prediction of stock prices. This is because volatility can tell us how volatile a stock is, which means we know how much its price changes in a day.

&emsp; &emsp; Supervised Learning

We used Long Short Term Memory (LSTM) and basic linear regression models on our dataset for supervised learning. We selected the LSTM model because of its feedback connections. This feature allows us more than single data points and is well-known for prediction problems. We also used the basic linear regression model to compare and contrast the differences between models. 
We evaluate the performance of our models through the visualizations generated. We train the model with a portion of our dataset and predict based on a small section. We determine the accuracy of our model by seeing whether the prediction trend is similar to our actual validation trend. 
Both LSTM and the basic linear regression models demonstrated significant overfitting. The prediction lines very closely matched the validation lines, which should not have happened because the trends were “unpredictable” to a certain extent. This could be due to a small dataset because of the 5-year restriction. 

# References


Roondiwala, M. (n.d.). Predicting stock prices using LSTM - ResearchGate. International Journal of Science and Research (IJSR). Retrieved June 9, 2022, from https://www.researchgate.net/profile/Murtaza-Roondiwala/publication/327967988_Predicting_Stock_Prices_Using_LSTM/links/5bafbe6692851ca9ed30ceb9/Predicting-Stock-Prices-Using-LSTM.pdf 

Xu, Y., Yang, C., Peng, S. et al. A hybrid two-stage financial stock forecasting algorithm based on clustering and ensemble learning. Appl Intell 50, 3852–3867 (2020). https://doi.org/10.1007/s10489-020-01766-5

Adusumilli, R. (2020, January 29). Predicting stock prices using a Keras LSTM model. Medium. Retrieved June 8, 2022, from https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233 
