<img src="https://ichef.bbci.co.uk/news/976/cpsprodpb/404F/production/_112236461_gettyimages-1209519827.jpg">


# Kaggle-COVID-tweet
 
This is a solution for [Kaggle NLP project](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification). The task is to classify sentiments in given tweets.

Twitter is a place to express thoughts and ideas and spread news and information, thus makes it a great source to track trends worldwide. Through the collected tweets, we can evaluate people's attitude toward Covid and pandemic through the trending keywords and tags through time. The estimator can be useful for identifying inappropriate tweets and stop toxic propaganda or evaluating strategies to promote social distancing using trend words.

Data has following columns:

|Column |Notes |
|:------|:-------|
UserName | User ID
ScreenName| User display name
Location | User location
TweetAt | Date
OriginalTweet | Tweets
Sentiment | Extremely negative, Negative, Neutral, Positive, Extremely positive

## Approach
Approach the problem in two ways:
1. Covid-tweets-sentiment_method1.ipynb approach the problem using one classifier and classify five sentiments.
2. Covid-tweets-sentiment_method2.ipynb approach the problem using three classifiers:
    - Sentiment classifier: classifies 3 classes: Positive (including Extremely Positive and Positive), Neutral, and Negative (Extremely Negative and Negative)
    - Extremely Positive classifier: identifies Extremely Positive from Positive, thus the model is binary classifier determines if input is extremely positive or not
    - Extremely Negative classifier: identifies Extremely Negative from Negative, thus the model is binary classifier determines if input is extremely negative or not
    
## Text Cleaning
Clean both the text and location through following step:
1. Remove links: remove links starting with http
2. Remove accounts: remove mentioned accounts except the most frequently mentioned accounts such as @realDonaldTrump, @Tesco, @BorisJohnson, etc.
3. Remove hashtags: remove hashtags including #covid* or #corona* because those  except the most popular hashtags
4. Remove special characters: special characters don't help identifying sentiment
5. Remove non-English characters
6. Remove stopwords: remove common words that don't help identifying sentiment
7. Stemming: replace word with its stem
8. Lemmatization: replace word with its lemma
9. Combine location and tweet content in one sentence
10. Lowercase

Here's what it's like before and after text cleaning:

|Original |Cleaned |
|:------|:-------|
@MeNyrbie @Phil_Gahan @Chrisitv https://t.co/iFz9FAn2Pa and https://t.co/xX6ghGFzCC and https://t.co/I2NlzdxNo8 | gb
advice Talk to your neighbours family to exchange phone numbers create contact list with phone numbers of neighbours schools employer chemist GP set up online shopping accounts if poss adequate supplies of regular meds but not over order | gb advice talk neighbours family exchange phone numbers create contact list phone numbers neighbours schools employer chemist gp set online store accounts poss adequate supplies regular meds order
Coronavirus Australia: Woolworths to give elderly, disabled dedicated shopping hours amid COVID-19 outbreak https://t.co/bInCA9Vp8P | australia woolworths give elderly disabled dedicated store hour amid outbreak
My food stock is not the only one which is empty...\r\r\n\r\r\nPLEASE, don't panic, THERE WILL BE ENOUGH FOOD FOR EVERYONE if you do not take more than you need. \r\r\nStay calm, stay safe.\r\r\n\r\r\n#COVID19france #COVID_19 #COVID19 #coronavirus #confinement #Confinementotal #ConfinementGeneral https://t.co/zrlG0Z520j | food stock empty please panic enough food everyone take need stay calm stay safe
Me, ready to go at supermarket during the #COVID19 outbreak.\r\r\n\r\r\nNot because I'm paranoid, but because my food stock is litteraly empty. The #coronavirus is a serious thing, but please, don't panic. It causes shortage...\r\r\n\r\r\n#CoronavirusFrance #restezchezvous #StayAtHome #confinement https://t.co/usmuaLq72n | ready go market outbreak paranoid food stock litteraly empty serious thing please panic causes shortage
As news of the regionÂs first confirmed COVID-19 case came out of Sullivan County last week, people flocked to area stores to purchase cleaning supplies, hand sanitizer, food, toilet paper and other goods, @Tim_Dodson reports https://t.co/cfXch7a2lU | news first confirmed case came sullivan county last week people flocked area store purchase cleaning supplies hand sanitizer food toiletpaper toiletpaper goods reports

## Modeling
Three models to train, tune, and evaluate:
1. Input -> Embedded -> Conv1D -> GlobalAveragePooling1D -> Dense -> Output
2. Input -> Embedded -> Bidirectional LSTM -> Dropout -> Dense -> Output
3. Input -> Embedded -> Bidirectional GRU -> Dropout -> Dense -> Output

## Performance:
- With one classifier: 73% accuracy
- With multiple classifiers: 66% accuracy

|Original Tweet | Cleaned Tweet | True Sentiment | Misclassified |
|:------|:-------|:-------|:--------|
| @DrTedros "We canÂt stop #COVID19 without protecting #healthworkers.Prices of surgical masks have increased six-fold, N95 respirators have more than trebled &amp; gowns cost twice as much"-@DrTedros #coronavirus | us stop without protecting prices surgical mask increase six fold respirators trebled amp gowns cost twice many | Neutral | Negative|
|HI TWITTER! I am a pharmacist. I sell hand sanitizer for a living! Or I do when any exists. Like masks, it is sold the fuck out everywhere. SHOULD YOU BE WORRIED? No. Use soap. SHOULD YOU VISIT TWENTY PHARMACIES LOOKING FOR THE LAST BOTTLE? No. Pharmacies are full of sick people.| hi twitter pharmacist sell hand sanitizer living exists like mask sold fuck everywhere worried use soap visit twenty pharmacies looking last bottle pharmacies full sick people|Extremely Negative|Negative|
|For those of you that think credit/debit is just as good as #bitcoin when it comes to combating #coronavirus, keep in mind that this is not a "contactless transaction."  How many people at your grocery store or gas station touch this keypad every day? https://t.co/WVq8bb9OlS|us think credit debit good comes combating keep mind contactless transaction many people goods store gas station touch keypad day|Extremely Positive|Positive
|Fellow Uni instructors! COVID-19 f2f class cancelations are  inconvenient! But remember you're salaried employees. Maybe call/email HR or your union &amp; demand to know how hourly workers (clerical, food-service, custodial, etc) will be  compensated during campus closures?|us fellow uni instructors class cancelations inconvenient remember salaried worker maybe call email hr union amp demand know hourly worker clerical food service custodial compensated campus closures|Negative|Positive|
Ok if #COVID2019 is nothing to panic about why is Italy imposing the biggest restrictions on the civilian population since WW2? How will the supermarkets be able to provide food if all the workers are told to stay at home? Same with any other Bussiness.|gb ok nothing panic italy imposing biggest restrictions civilian population since ww market able provide food worker told stay home bussiness|Positive|Negative

## Conclusion
When reviewing the data set, some tweets' label are quite questionable. And some of the predicted sentiments actually makes more sense than the true label. For improvement, consider re-evaluate the data set labels and adjust text cleaning technique accordingly. In addition, clarifying keywords or criteria that make a tweet "extremely" positive or negative helps us adjust our model to be more sensitive.
