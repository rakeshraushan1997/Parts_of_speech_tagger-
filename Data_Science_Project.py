
import numpy as np
import random
from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from collections import namedtuple, OrderedDict

Sentence = namedtuple("Sentence", "words tags")

def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)

def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))

def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)

class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):  
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())

data = Dataset(r"D:\Programming\Data_Science_Project\tags-universal.txt", r"D:\Programming\Data_Science_Project\brown-universal.txt", train_test_split=0.7)

assert len(data) == len(data.training_set) + len(data.testing_set), \
       "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"

def pair_counts(tags, words):
    d = defaultdict(lambda: defaultdict(int))
    for tag, word in zip(tags, words):
        d[tag][word] += 1
        
    return d

def unigram_counts(sequences):

    return Counter(sequences)

def bigram_counts(sequences):

    d = Counter(sequences)
    return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]         #Collects the tags
o = [(tags[i],tags[i+1]) for i in range(0,len(tags)-2,2)]           #collects pairs of tags
tag_bigrams = bigram_counts(o) 

def starting_counts(sequences):
    
    d = Counter(sequences)
    return d

def ending_counts(sequences):
    
    d = Counter(sequences)
    return d

#Model Accuracy Evaluation

def accuracy(X, Y, model):
    
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions


#IMPLEMENTATION: Basic HMM Tagger
basic_model = HiddenMarkovModel(name="base-hmm-tagger")

tags = [tag for i, (word, tag) in enumerate(data.stream())]            # tags in whole corpus
words = [word for i, (word, tag) in enumerate(data.stream())]          # words in whole corpus

tags_count=unigram_counts(tags)                                        #Counts the no. of tags in whole corpus "tag_unigrams"
tag_words_count=pair_counts(tags,words)                     #Give count of a particular tags with word appeared in sentence

starting_tag_list=[i[0] for i in data.Y]
ending_tag_list=[i[-1] for i in data.Y]

starting_tag_count=starting_counts(starting_tag_list)            #the number of times a tag occured at the start
ending_tag_count=ending_counts(ending_tag_list)                  #the number of times a tag occured at the end



to_pass_states = []
for tag, words_dict in tag_words_count.items():
    total = float(sum(words_dict.values()))
    distribution = {word: count/total for word, count in words_dict.items()}
    tag_emissions = DiscreteDistribution(distribution)
    tag_state = State(tag_emissions, name=tag)
    to_pass_states.append(tag_state)


basic_model.add_states()    
    

start_prob={}

for tag in tags:
    start_prob[tag]=starting_tag_count[tag]/tags_count[tag]

for tag_state in to_pass_states :
    basic_model.add_transition(basic_model.start,tag_state,start_prob[tag_state.name])    

end_prob={}

for tag in tags:
    end_prob[tag]=ending_tag_count[tag]/tags_count[tag]
for tag_state in to_pass_states :
    basic_model.add_transition(tag_state,basic_model.end,end_prob[tag_state.name])
    


transition_prob_pair={}

for key in tag_bigrams.keys():
    transition_prob_pair[key]=tag_bigrams.get(key)/tags_count[key[0]]
for tag_state in to_pass_states :
    for next_tag_state in to_pass_states :
        basic_model.add_transition(tag_state,next_tag_state,transition_prob_pair[(tag_state.name,next_tag_state.name)])

basic_model.bake()

#Prediction Making

def replace_unknown(sequence):
    
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]

#Visual inspection of model accuracy

for key in data.testing_set.keys[2:3]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")

#Overall accuracy of our model
hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

sentence = input("Enter your sentence : ")
formate=list(sentence.split())
print(simplify_decoding(formate, basic_model))
