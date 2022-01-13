import re


class Tweet:

    def __init__(self, Id, tweet, following, followers, actions, is_retweet, location, Type):
        self.Id = int(Id)
        self.tweet = str(tweet)
        self.following = int(following)
        self.followers = int(followers)
        self.actions = float(actions)
        self.is_retweet = int(is_retweet)
        self.location = str(location)
        self.Type = str(Type)
        pass

    def getid(self):
        return self.Id

    def setid(self, Id):
        self.Id = Id

    def gettweet(self):
        return self.tweet

    def settweet(self, tweet):
        self.tweet = tweet

    def getfollowing(self):
        return self.following

    def setfollowing(self, following):
        self.following = following

    def getfollowers(self):
        return self.followers

    def setfollowers(self, followers):
        self.followers = followers

    def getactions(self):
        return self.actions

    def setactions(self, actions):
        self.actions = actions

    def getisretweet(self):
        return self.is_retweet

    def setisretweet(self, is_retweet):
        self.is_retweet = is_retweet

    def getlocation(self):
        return self.location

    def setlocation(self, location):
        self.location = location

    def getType(self):
        return self.Type

    def setType(self, Type):
        self.Type = Type

    def __str__(self):
        print("\n\nTweet:", self.Id,
              "\nbody: ", self.tweet,
              "\nfollowing: ", self.following,
              "\nfollowers: ", self.followers,
              "\nactions: ", self.actions,
              "\nis retweeted: ", self.is_retweet,
              "\nlocation: ", self.location,
              "\ntype: ", self.Type)


class Features:

    def __init__(self, tweet):
        self.tweet = tweet
        self.no_hashtag = self.calculate_no_hashtag()
        self.no_usermention = self.calculate_no_usermention()
        self.no_url = self.calculate_no_url()
        self.no_char = self.calculate_no_char()
        self.no_digit = self.calculate_no_digit()
        self.no_word = self.calculate_no_word()
        self.no_actions = self.tweet.getactions()
        self.no_follower = self.tweet.getfollowers()
        self.no_following = self.tweet.getfollowing()
        self.reputation = self.calculate_reputation()

        pass

    def calculate_no_hashtag(self):
        tweet_str = self.tweet.gettweet()
        count = 0
        for i in tweet_str:
            if i == '#':
                count += 1
        return count

    def calculate_no_usermention(self):
        tweet_str = self.tweet.gettweet()
        count = 0
        for s in tweet_str:
            if s == '@':
                count += 1
        return count
        pass

    def calculate_no_url(self):
        urls = re.findall(
            r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
            self.tweet.gettweet())  # regex expression to extract URLs
        return len(urls)

    def calculate_no_char(self):
        return len(self.tweet.gettweet())

    def calculate_no_digit(self):
        tweet_str = self.tweet.gettweet()
        count = 0
        for s in tweet_str:
            if s.isnumeric():
                count += 1
        return count

    def calculate_no_word(self):
        tweet_str = self.tweet.gettweet()
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

        f_v = [self.no_word, self.no_char, self.no_digit, self.no_char,
               self.no_hashtag, self.no_usermention, self.no_url,
               self.no_following, self.no_follower, self.reputation,
               self.no_actions]

        if self.tweet.getType() == 'Spam':
            label = 1
        else:
            label = 0

        return [f_v, label]
