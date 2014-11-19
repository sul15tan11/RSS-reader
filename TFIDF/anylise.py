import numpy
from textblob import TextBlob


wiki = TextBlob("Python is a high-level, general-purpose programming language. so it is a proper language")
print wiki.tags
print wiki.noun_phrases

testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
print testimonial.sentiment
print wiki.words
print wiki.sentences

for sentence in wiki.sentences:
    print(sentence.sentiment)
    
from textblob import Word
from textblob.wordnet import VERB

word = Word("octopus")
print word.synsets
print Word("octopus").definitions

print wiki.word_counts['language']

print wiki.json


print "----------------------------------- SECOND PART ----------------------------";
#--------------------

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
blob.tags           # [(u'The', u'DT'), (u'titular', u'JJ'),
                    #  (u'threat', u'NN'), (u'of', u'IN'), ...]

blob.noun_phrases   # WordList(['titular threat', 'blob',
                    #            'ultimate movie monster',
                    #            'amoeba-like mass', ...])

for sentence in blob.sentences:
    print"aaa",(sentence.sentiment.polarity)
# 0.060
# -0.341

print " ------------------------- TRANLSLATED TO ESPANIOLA ---------------"
print blob.translate(to="fr")  # 'La amenaza titular de The Blob...'
print " ------------------------- END TRANLSLATION ---------------"



print "----------------------------------- THIRD PART ----------------------------";
import numpy
from sklearn.feature_extraction.text import CountVectorizer

train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.",
"We can see the shining sun, the bright sun.")

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(train_set)
print "Vocabulary:", count_vectorizer.vocabular

# Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}

freq_term_matrix = count_vectorizer.transform(test_set)
print freq_term_matrix.todense()

#[[0 1 1 1]
#[0 2 1 0]]

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

print "IDF:", tfidf.idf_

# IDF: [ 0.69314718 -0.40546511 -0.40546511  0.        ]
tf_idf_matrix = tfidf.transform(freq_term_matrix)
print tf_idf_matrix.todense()