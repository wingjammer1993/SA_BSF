import re
import sys
import math
import nltk
import io


# Gives the number of reviews in the input document
# marker is the start of review with ID-

def count_n_c(input_document):
    ip = io.open(input_document, 'r', encoding="utf8")
    all_tokens = ip.read()
    id_tokens = re.findall("ID-", all_tokens)
    count = len(id_tokens)
    return count


# Gives the log prior probability
# N_c/N_doc ( number of reviews in class /  number of total reviews )

def log_prior_probability(input_pos, input_neg):
    log_prior_prob = {}
    pos = count_n_c(input_pos)
    neg = count_n_c(input_neg)
    n_doc = pos + neg
    num1 = pos/float(n_doc)
    num2 = neg/float(n_doc)
    log_prior_prob['input_pos'] = math.log10(num1)
    log_prior_prob['input_neg'] = math.log10(num2)
    return log_prior_prob


# Gives the vocabulary in input document

def vocabulary(input_document):
    ip = io.open(input_document, 'r', encoding="utf8")
    all_tokens = ip.read()
    tokens = nltk.wordpunct_tokenize(all_tokens)
    words = [w.lower() for w in tokens]
    vocab = (sorted(set(words)))
    return vocab


# Gives the total vocabulary of all classes

def get_vocab_training(input_pos, input_neg):
    vocab_pos = vocabulary(input_pos)
    vocab_neg = vocabulary(input_neg)
    v = list(vocab_pos)
    v.extend(list(vocab_neg))
    v = (sorted(set(v)))
    v = clean_vocab(v)
    return v


# Gives the bag of words in input_documents, returns iterable FreqDist

def big_doc_c(input_document):
    ip = io.open(input_document, 'r', encoding="utf8")
    all_tokens = ip.read()
    tokens = nltk.wordpunct_tokenize(all_tokens)
    words = [w.lower() for w in tokens]
    words = clean_vocab(words)
    freq_dist = nltk.FreqDist(words)
    return freq_dist


# This method removes numbers,punctuations,one word characters from the vocabulary

def clean_vocab(vocab):
    clean_vocabulary = []
    for word in vocab:
        if re.match("^[a-zA-Z]+$", word):
            clean_vocabulary.append(word)
    return clean_vocabulary


# Train the Naive-Bayes, It will return log likelihoods in the form of dictionary
# word : log_prob_positive, log_prob_negative

def train_naive_bayes(input_pos, input_neg, smoothing_factor):
    log_likelihood = {}
    vocab_total = get_vocab_training(input_pos, input_neg)
    freq_dist_pos = big_doc_c(training_pos)
    freq_dist_neg = big_doc_c(training_neg)
    count_total_pos = freq_dist_pos.N()
    count_total_neg = freq_dist_neg.N()
    v = len(vocab_total)
    v_alpha = smoothing_factor * v

    for word in vocab_total:
        log_probs = []
        count_pos = freq_dist_pos[word]
        count_neg = freq_dist_neg[word]
        prob_pos = (count_pos + smoothing_factor) / float(count_total_pos + v_alpha)
        prob_neg = (count_neg + smoothing_factor) / float(count_total_neg + v_alpha)
        log_prob_pos = math.log10(prob_pos)
        log_prob_neg = math.log10(prob_neg)
        log_probs.append(log_prob_pos)
        log_probs.append(log_prob_neg)
        log_likelihood[word] = list(log_probs)
    return log_likelihood


# Test the data in test_doc based on previous training

def test_naive_bayes(input_pos, input_neg, test_doc, smoothing_factor):

    log_prior = log_prior_probability(input_pos, input_neg)
    vocab_training = get_vocab_training(input_pos, input_neg)
    log_likelihood = train_naive_bayes(input_pos, input_neg, smoothing_factor)
    output = {}
    test = io.open(test_doc, 'r', encoding="utf8")
    for line in test.readlines():
        if line != '\n':
            log_class = dict(log_prior)
            tokens = nltk.wordpunct_tokenize(line)
            words = [w.lower() for w in tokens]
            for word in words:
                if word in vocab_training:
                    log_class['input_pos'] = log_class['input_pos'] + float(log_likelihood[word][0])
                    log_class['input_neg'] = log_class['input_neg'] + float(log_likelihood[word][-1])
            if log_class['input_pos'] > log_class['input_neg']:
                output[line.split(None, 1)[0]] = 'POS'
            else:
                output[line.split(None, 1)[0]] = 'NEG'
    return output


# Create gold_std of all reviews

def get_gold_std(train_pos, train_neg):
    gold_set = {}
    test = io.open(train_pos, 'r', encoding="utf8")
    for line in test.readlines():
        gold_set[line.split(None, 1)[0]] = 'POS'
    test = io.open(train_neg, 'r', encoding="utf8")
    for line in test.readlines():
        gold_set[line.split(None, 1)[0]] = 'NEG'
    return gold_set


# Give the accuracy

def give_accuracy(train_pos, train_neg, test_output):
    gold_set = get_gold_std(train_pos, train_neg)
    count = 0
    for elem in test_output:
        if test_output[elem] == gold_set[elem]:
            count = count + 1
    return count / float(len(test_output))


if __name__ == "__main__":

    training_pos = r'positive.txt'
    training_neg = r'negative.txt'
    gold_pos = r'hotelPosT-train.txt'
    gold_neg = r'hotelNegT-train.txt'
    dev_set = r'test_doc.txt'
    verdict = test_naive_bayes(training_pos, training_neg, dev_set, 1)
    accuracy = give_accuracy(gold_pos, gold_neg, verdict)
    print(verdict)
    print(accuracy)

    #   training_pos = sys.argv[1]  # r'hotelNegT-train.txt' #
    #   training_neg = sys.argv[2]  # r'hotelPosT-train.txt'  #






