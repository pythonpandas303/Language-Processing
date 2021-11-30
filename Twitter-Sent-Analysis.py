import sys
import jsonlines

def readTwitterData(twitterDataFile):
    tweets = []
    with jsonlines.open(twitterDataFile) as infile:
        reader = jsonlines.Reader(infile)
        for line in reader:
            tweets.append(line)
    print(type(tweets[0]))
    return tweets


def readSentimentData(sentimentDataFile):
    sentimentfile = open(sentimentDataFile, "r")	
    scores = {}  									
    for line in sentimentfile:                     
        word, score = line.split("\t")  			
        scores[word] = int(score)   				
    sentimentfile.close()							
    return scores								


def main():
    if len(sys.argv) > 1:
        scores_file  = sys.argv[1]         
        tweets_file = sys.argv[2]           
    else:
        scores_file = input('Enter AFIN file name: ')
        tweets_file = input('Enter tweet data: ')

    scores = readSentimentData(scores_file)
    tweets = readTwitterData(tweets_file)
    sentiments = {"-5": 0, "-4": 0, "-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}   

    for tweet in tweets:                           
        if 'text' in tweet:
            tweetWords = tweet['text'].split()                 
            for word in tweetWords:                     
                word = word.strip('?:!.,;"!@')         
                word = word.replace("\n", "")           
                if word in scores.keys():               
                    score = scores[word]                
                    sentiments[str(score)] += 1         

# print number of sentiments in each category
    print("-5 sentiments ", sentiments["-5"])
    print("-4 sentiments ", sentiments["-4"])
    print("-3 sentiments ", sentiments["-3"])
    print("-2 sentiments ", sentiments["-2"])
    print("-1 sentiments ", sentiments["-1"])
    print(" 0 sentiments ", sentiments["0"])
    print(" 1 sentiments ", sentiments["1"])
    print(" 2 sentiments ", sentiments["2"])
    print(" 3 sentiments ", sentiments["3"])
    print(" 4 sentiments ", sentiments["4"])
    print(" 5 sentiments ", sentiments["5"])


if __name__ == '__main__':
    main()

