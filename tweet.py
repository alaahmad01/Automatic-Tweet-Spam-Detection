import re
import pandas as pd
from nlp import text_learning


def extract_features(filename):
    """
    function to extract the features of a tweet in a csv file using text learning and feature extraction methods
    """
    features = []

    df = pd.read_csv(filename)

    df = df.dropna(axis=0, subset=['following', 'followers', 'is_retweet', 'Tweet', 'actions', 'Type'])  # remove nulls
    df = df.drop_duplicates(keep='first')  # remove duplicates

    # nlp_results trains only on the third of the data set and predicts the other two thirds
    # the prediction results will act as an attribute in the feature vector
    nlp_results = text_learning(df.Tweet, df.Type)

    # set for feature extraction
    df2 = df.iloc[int(df.shape[0] * (1/3)):]

    for row, res in zip(df2.itertuples(), nlp_results):
        features.append(Features(Tweet(int(row.Id),
                                       str(row.Tweet),
                                       int(row.following),
                                       int(row.followers),
                                       int(row.actions),
                                       int(row.is_retweet),
                                       str(row.location),
                                       str(row.Type)
                                       ), res
                                 ).assemble_vector())

    return features


class Tweet:

    def __init__(self, Id, tweet, following, followers, actions, is_retweet, location, Type):
        self.Id = int(Id)
        self.tweet = str(tweet)
        self.following = int(following)
        self.followers = int(followers)
        self.actions = int(actions)
        self.is_retweet = int(is_retweet)
        self.location = str(location)
        self.Type = str(Type)
        pass

    def getTweet(self):
        return self.tweet

    def getFollowing(self):
        return self.following

    def getFollowers(self):
        return self.followers

    def getActions(self):
        return self.actions

    def getType(self):
        return self.Type

    def __str__(self):
        st = ''
        string = "\n\nTweet:", self.Id, \
                 "\nbody: ", self.tweet, \
                 "\nfollowing: ", self.following, \
                 "\nfollowers: ", self.followers, \
                 "\nactions: ", self.actions, \
                 "\nis retweeted: ", self.is_retweet, \
                 "\nlocation: ", self.location, \
                 "\ntype: ", self.Type, "\n"
        for s in string:
            st += str(s)
        return st


class Features:

    def __init__(self, tweet, nlp_res):
        self.tweet = tweet
        self.nlp_result = nlp_res
        self.no_hashtag = self.calculate_no_hashtag()
        self.no_usermention = self.calculate_no_usermention()
        self.no_url = self.calculate_no_url()
        self.no_char = self.calculate_no_char()
        self.no_digit = self.calculate_no_digit()
        self.no_word = self.calculate_no_word()
        self.no_actions = self.tweet.getActions()
        self.no_follower = self.tweet.getFollowers()
        self.no_following = self.tweet.getFollowing()
        self.reputation = self.calculate_reputation()
        self.url_word = self.calculate_url_word()
        self.hashtag_word = self.calculate_hashtag_word()

        pass

    def clean_tweet(self):
        return re.sub('[^\w\s]', '', self.tweet.getTweet())
        pass

    def calculate_no_hashtag(self):
        tweet_str = self.tweet.getTweet()
        count = 0
        for i in tweet_str:
            if i == '#':
                count += 1
        return count

    def calculate_no_usermention(self):
        tweet_str = self.tweet.getTweet()
        count = 0
        for s in tweet_str:
            if s == '@':
                count += 1
        return count
        pass

    def calculate_no_url(self):
        urls = re.findall(
            r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
            self.tweet.getTweet())  # regex expression to extract URLs
        return len(urls)

    def calculate_no_char(self):
        return len(self.tweet.getTweet())

    def calculate_no_digit(self):
        tweet_str = self.tweet.getTweet()
        count = 0
        for s in tweet_str:
            if s.isnumeric():
                count += 1
        return count

    def calculate_no_word(self):
        tweet_str = self.tweet.getTweet()
        count = 0
        for s in tweet_str:
            if s == ' ' or s == '\n' or s == '\t':
                count += 1
        return count

    def calculate_hashtag_word(self):
        return self.no_hashtag / (self.no_word + 1)

    def calculate_url_word(self):
        return self.no_url / (self.no_word + 1)

    def calculate_reputation(self):
        return self.no_following / (self.no_follower + 1)

    def assemble_vector(self):

        f_v = [self.no_word, self.no_char, self.no_digit,
               self.no_hashtag, self.no_usermention, self.no_url,
               self.no_following, self.no_follower, self.reputation,
               self.no_actions, self.url_word, self.hashtag_word,
               self.nlp_result]

        return [f_v, 1 if self.tweet.getType() == 'Spam' else 0]
