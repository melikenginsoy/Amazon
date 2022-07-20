#############################################
# Rating Product & Sorting Reviews in Amazon
#############################################


##################
# Business Problem
##################

"""
One of the most important problems in e-commerce is the correct calculation of
the points given to the products after sales.
Another problem is the correct ordering of the comments given to the products.
With the solution of these problems, the e-commerce site and the sellers will increase
their sales, while the customers will complete the purchasing journey without any problems.
"""


#############################
# Dataset Story and Variables
#############################
"""
This dataset, which includes Amazon product data, includes product categories and various metadata.
The product with the most reviews in category X has user ratings and reviews.

12 Variables    4915 Observations     71.9 MB

# reviewerID            :User ID
# asin                  :Product ID
# reviewerName          :User name
# helpful               :Useful review rating
# reviewText            :Review text
# overall               :Product rating
# summary               :Review summary
# unixReviewTime        :Review time
# reviewTime            :Review time Raw
# day_diff              :Number of days since review
# helpful_yes           :The number of times the review was found useful
# total_vote            :Number of votes given to the review
"""


###############################
# Preparing and Analyzing Data
###############################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Projects/Amazon/amazon_review.csv")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


"""
##################### Shape #####################
(4915, 12)
##################### Types #####################
reviewerID         object
asin               object
reviewerName       object
helpful            object
reviewText         object
overall           float64
summary            object
unixReviewTime      int64
reviewTime         object
day_diff            int64
helpful_yes         int64
total_vote          int64
dtype: object
##################### Head #####################
       reviewerID        asin  reviewerName helpful                                         reviewText  overall                                 summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
0  A3SBTW3WS4IQSN  **********            NaN  [0, 0]                                         No issues.  4.00000                              Four Stars      1406073600  2014-07-23       138            0           0
1  A18K1ODH1I2MVB  **********           0mie  [0, 0]  Purchased this for my device, it worked as adv...  5.00000                           MOAR SPACE!!!      1382659200  2013-10-25       409            0           0
2  A2FII3I2MBMUIA  **********            1K3  [0, 0]  it works as expected. I should have sprung for...  4.00000               nothing to really say....      1356220800  2012-12-23       715            0           0
3   A3H99DFEG68SR  **********            1m2  [0, 0]  This think has worked out great.Had a diff. br...  5.00000  Great buy at this price!!!  *** UPDATE      1384992000  2013-11-21       382            0           0
4  A375ZM4U047O79  **********   2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...  5.00000                        best deal around      1373673600  2013-07-13       513            0           0
##################### Tail #####################
          reviewerID        asin reviewerName helpful                                         reviewText  overall                        summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
4910  A2LBMKXRM5H2W9  **********        ZM "J"  [0, 0]  I bought this Sandisk 16GB Class 10 to use wit...  1.00000       Do not waste your money.      1374537600  2013-07-23       503            0           0
4911   ALGDLRUI1ZPCS  **********            Zo  [0, 0]  Used this for extending the capabilities of my...  5.00000                    Great item!      1377129600  2013-08-22       473            0           0
4912  A2MR1NI0ENW2AD  **********     Z S Liske  [0, 0]  Great card that is very fast and reliable. It ...  5.00000  Fast and reliable memory card      1396224000  2014-03-31       252            0           0
4913  A37E6P3DSO9QJD  **********      Z Taylor  [0, 0]  Good amount of space for the stuff I want to d...  5.00000              Great little card      1379289600  2013-09-16       448            0           0
4914   A8KGFTFQ86IBR  **********           Zza  [0, 0]  I've heard bad things about this 64gb Micro SD...  5.00000                So far so good.      1388620800  2014-02-01       310            0           0
##################### NA #####################
reviewerID        0
asin              0
reviewerName      1
helpful           0
reviewText        1
overall           0
summary           0
unixReviewTime    0
reviewTime        0
day_diff          0
helpful_yes       0
total_vote        0
dtype: int64
##################### Quantiles #####################
                    count             mean            std              min               0%               5%              50%              95%              99%             100%              max
overall        4915.00000          4.58759        0.99685          1.00000          1.00000          2.00000          5.00000          5.00000          5.00000          5.00000          5.00000
unixReviewTime 4915.00000 1379465001.66836 15818574.32275 1339200000.00000 1339200000.00000 1354492800.00000 1381276800.00000 1403308800.00000 1404950400.00000 1406073600.00000 1406073600.00000
day_diff       4915.00000        437.36704      209.43987          1.00000          1.00000         98.00000        431.00000        748.00000        943.00000       1064.00000       1064.00000
helpful_yes    4915.00000          1.31109       41.61916          0.00000          0.00000          0.00000          0.00000          1.00000          3.00000       1952.00000       1952.00000
total_vote     4915.00000          1.52146       44.12309          0.00000          0.00000          0.00000          0.00000          1.00000          4.00000       2020.00000       2020.00000
"""


##################
# Rating Product
##################

df["overall"].value_counts()
"""
5.00000    3922
4.00000     527
1.00000     244
3.00000     142
"""

# Average rating of the product

df["overall"].mean()
# 4.587589013224822

# Converting reviewTime to datetime

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df.info()

# Expression of reviews in days

current_date = df["reviewTime"].max()

df["days"] = (current_date - df["reviewTime"]).dt.days

df.head()

# Examining the quartiles of the days variable

sns.boxplot(x=df["days"]);

df["days"].describe()
"""
count   4915.00000
mean     436.36704
std      209.43987
min        0.00000
25%      280.00000
50%      430.00000
75%      600.00000
max     1063.00000
"""

q1 = df["days"].quantile(q=0.25)   # 280.0

q2 = df["days"].quantile(q=0.5)    # 430.0

q3 = df["days"].quantile(q=0.75)   # 600.0

# Creation of time zones

df.loc[df["days"] <= q1, "overall"].mean()            # 4.6957928802588995

df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean()     # 4.636140637775961

df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean()    # 4.571661237785016

df.loc[(df["days"] > q3), "overall"].mean()   # 4.4462540716612375

# time based weighted average rating

def time_based_weighted_average(dataframe, w1=28, w2=27, w3=23, w4=22):
    return dataframe.loc[df["days"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > q3), "overall"].mean() * w4 / 100


time_based_weighted_average(df)    # 4.596237959128027


##################
# Sorting Reviews
##################

# Define the helpful_no variable

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# score_pos_neg_diff

def score_up_down_diff(yes, no):
    return yes - no


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)


# score_average_rating

def score_average_rating(yes, no):
    if yes + no == 0:
        return 0
    return yes / (yes + no)


df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


# wilson_lower_bound

def wilson_lower_bound(yes, no, confidence=0.95):
    """
    Wilson Lower Bound Score calculate

    - The lower limit of the confidence interval to be calculated for
      the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    - Note:
    If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and
    can be made to conform to Bernoulli.
    This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    yes: int
        up count
    no: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = yes + no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


df.sort_values("wilson_lower_bound", ascending=False).head(20)
"""
         overall    days    helpful_yes    helpful_no    score_pos_neg_diff  score_average_rating  wilson_lower_bound              
2031     5.00000     701           1952            68                  1884               0.96634             0.95754
3449     5.00000     802           1428            77                  1351               0.94884             0.93652
4212     1.00000     578           1568           126                  1442               0.92562             0.91214
317      1.00000    1032            422            73                   349               0.85253             0.81858
4672     5.00000     157             45             4                    41               0.91837             0.80811
1835     5.00000     282             60             8                    52               0.88235             0.78465
3981     5.00000     776            112            27                    85               0.80576             0.73214
3807     3.00000     648             22             3                    19               0.88000             0.70044
4306     5.00000     822             51            14                    37               0.78462             0.67033
4596     1.00000     806             82            27                    55               0.75229             0.66359
315      5.00000     846             38            10                    28               0.79167             0.65741
1465     4.00000     237              7             0                     7               1.00000             0.64567
1609     5.00000     256              7             0                     7               1.00000             0.64567
4302     5.00000     261             14             2                    12               0.87500             0.63977
4072     5.00000     758              6             0                     6               1.00000             0.60967
1072     5.00000     941              5             0                     5               1.00000             0.56552
2583     5.00000     488              5             0                     5               1.00000             0.56552
121      5.00000     942              5             0                     5               1.00000             0.56552
1142     5.00000     306              5             0                     5               1.00000             0.56552
1753     5.00000     776              5             0                     5               1.00000             0.56552
"""