# ================================ Contributers ================================
"""
Nathaniel Maymon
"""

# ================================ imports ================================
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation
from math import log10, sqrt
from pandas import read_csv
from heapq import heappop, heapify
from tqdm import tqdm

# ================================ General Variables ================================

LANGUAGE = 'english'

# In our case, NUM_OF_RESULTS = 5 = K. Thus, O(k)=O(1)
NUM_OF_RESULTS = 5

# tweets.csv size is 27346 rows
DOCS_TO_LOAD = 1000

normalize_values = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    '10': 'ten',
    '$': 'dollar',
    '%': 'percent',
    '&': '',
    '#': 'hashtag'
}

# ================================ Functions ================================

def my_tokenization_function(document_text):
    ''' Decode, tokenize & normalize a document. Returns a list of a documents tokens. '''

    document_text = document_text.encode('utf-8')
    document_text = document_text.decode('unicode-escape')

    stop_words = set(list(punctuation) + stopwords.words(LANGUAGE))
    ps = PorterStemmer()

    # Tokenize
    terms = word_tokenize(document_text)

    # Normalize
    terms = [term if term not in normalize_values else normalize_values[term] for term in terms]

    # Remove stops and punctuations
    terms = [term for term in terms if term not in stop_words]

    # Stem
    terms = [ps.stem(term.lower()) for term in terms]

    return terms


class IREngine(object):
    def __init__(self):
        self.documents = {}
        self.inverted_index = {}
        self._load_engine()

    def _load_engine(self):
        ''' Load corpos from tweets.csv and intialize texts as documents. '''
    
        corpus = read_csv('tweets.csv').head(DOCS_TO_LOAD)
        tweets = corpus['text']

        # Tokenize-normalize every tweet and initialize as a indexed document
        for i in tqdm(range(0, DOCS_TO_LOAD)):
            self.__index(i, tweets[i], my_tokenization_function)


    def __index(self, idx, document_text, my_tokenization_function):
        ''' Initialize each document in a dictionary as a tuple containing the original text and a dictionary to represent its TF. '''
       
        self.documents[idx] = (document_text, {})
        doc_tokenz = my_tokenization_function(document_text)

        # Update inverted index with new documents terms
        self.__rebuild_index(doc_tokenz, idx)
        del doc_tokenz

        # Weight the new documents TF
        self.__weight_tf(self.documents[idx][1])

    def __rebuild_index(self, terms, idx):
        ''' Rebuild inverted_index by updating via a documents list of terms and ID. '''

        for term in terms:
            # Term and ID already exsits in inverted_index
            if (term in self.inverted_index) and (idx not in self.inverted_index[term]):
                self.inverted_index[term].append(idx)
            # New term in inverted_index
            else:
                self.inverted_index[term] = [idx]
            # Update TF for document
            self._set_condensed_vector(self.documents[idx][1], term)

    def _set_condensed_vector(self, doc, term):
        ''' Update TF for a given document '''
        if term in doc:
            doc[term] += 1
        # New occurrence of a term in the document that exsits in corpus
        elif term in self.inverted_index:
            doc[term] = 1

    def __weight_tf(self, doc):
        ''' Calculate the weighted TF for a term in a documents. W(tf) = 1+log10(tf). '''
        doc = {term: 1 + log10(doc[term]) for term in doc}    

    def _get_top_scores(self, scores):
        ''' 
            Recieves a list of cosine similarity scores as tuples containing a document ID and its score with the query.
            scores = [(score_1, ID_1), ..., (score_n, ID_n)].
            Returns a reverse sorted list of tuples with top K scores, where K = NUMBER_OF_RESULTS.
        '''
        res = []

        # Transform scores into a heap. O(n)
        heapify(scores)

        # Populate res list with top K results. O(klog2(n))
        size = NUM_OF_RESULTS if len(scores) >= NUM_OF_RESULTS else len(scores)

        for i in list(range(0, size)):
            score, idx = heappop(scores)
            res.append((idx, -score))

        print(res)
        return res

    def _cosine_similarity_scores(self, query, docs, idx):
        ''' 
            Creates IDF weighted condensed vectors from query and documents relevant to the query.
            Then calculates cosine similarity score of each vector with the query vector and stores
            it in a list(scores).
            Finally, returns a list of the of top K scores.

            Parameters:
                query: a dictionary containg the weighted TF for a query.
                docs:  a list of documents.
                idx:   Optional. Index of a document to skip in case of a "more like this document" query. type: None or int.

            Returns:
                top_docs: A list of K tuples containing the ids with the highest cosine similarity scores.
        '''

        vectors = {doc: {} for doc in docs if doc != idx}
        invidx = self.inverted_index
        scores = []

        # Construct a dictionary of TF-IDF weighted (condensed) vectors for query and docs
        query = {term: query[term] * (log10(len(self.documents) / len(invidx[term]))) for term in query}
        for idx in vectors:
            for term in self.documents[idx][1]:
                idf = log10(len(self.documents) / len(invidx[term]))
                vectors[idx][term] = self.documents[idx][1][term] * idf

        # Calculate squared sum of query vector
        magnitude_q = 0
        for term in query:
            magnitude_q += query[term]**2

        # Calculate the cosine similarity score between each vector and the query vector
        for idx in vectors:
            dot_product = 0
            magnitude_idx = 0

            # Calculate squared sum of document vector
            for term in vectors[idx]:
                magnitude_idx += vectors[idx][term]**2

            normals = sqrt(magnitude_q * magnitude_idx)

            # Calculate the Dot Product between query and document vectors
            for term in query:
                if term in vectors[idx]:
                    dot_product += query[term] * vectors[idx][term]

            cosine_score = dot_product / normals

            scores.append((-cosine_score, idx))

        top_docs = self._get_top_scores(scores)

        return top_docs

    def _get_scores(self, q_tf, idx):
        ''' 
            Recieves a dictionary containing a querys TF, and optionally an index of a document to skip.
            Retrieves all indecies that are relevant to the query via the inverted index.
            Calculates and returns the top K cosine similarity scores between relevant documents and query.

            Parameters:
                q_tf: A dictionary containg the weighted TF for a query.
                idx:  Optional. Index of a document to skip in case of a "more like this document" query. type: None or int.

            Returns:
                top_docs: A list of K tuples containg the id's with the highest cosine similarity scores.
        '''

        # Retrieve all relevant documents for the query
        docs = self.union(q_tf.keys())

        # If no document indices were retrieved, skip entire scoring process
        if not docs:
            top_docs = docs
        else:
            top_docs = self._cosine_similarity_scores(q_tf, docs, idx)
        
        return top_docs

    def intersection(self, term1, term2):
        ''' Returns a list of indicies from an 'And' query between 2 terms. '''

        # Both terms are in corpus
        if term1 in self.inverted_index and term2 in self.inverted_index:
            i = 0
            j = 0
            res = []
            lst1 = self.inverted_index[term1]
            lst2 = self.inverted_index[term2]

            # Sort term lists (unnecessary due to inverted_index's initialization).
            # Though it's important for lists to be sorted.
            #lst1.sort()
            #lst2.sort()

            # Assert the shorter list between terms
            shortest = lst1 if len(lst1) <= len(lst2) else lst2
            longest = lst2 if len(lst1) <= len(lst2) else lst1

            # Iterate over both lists and and append any intersection into res list.
            while (i < len(shortest) and j < len(longest)):
                if shortest[i] == longest[j]:
                    res.append(shortest[i])
                    i += 1
                    j += 1

                elif shortest[i] > longest[j]:
                    j += 1

                else:
                    i += 1

            return res
        else:
            return []

    def union(self, query):
        ''' Returns a unique list of indicies from an 'Or' query between 1-n terms. '''

        res = []

        # Fetch indices for each term in query and union to a res list.
        for term in query:
            if term in self.inverted_index:
                res += res+self.inverted_index[term]

        res = list(set(res))

        return res
    
    def process_query(self, query):
        ''' Transforms an open query into a dictionary of weighted TFs and sends it to cosine similarity scoring. '''

        # Tokenize-normalize query
        query_tokenz = my_tokenization_function(query)
        query_tf = {}

        # Transform query into a condensed vector (only contains terms that exist in corpus)
        for token in query_tokenz:
            self._set_condensed_vector(query_tf, token)
        self.__weight_tf(query_tf)
        del query_tokenz

        return self._get_scores(query_tf, None)

    def more_like(self, idx):
        ''' Sets the query to the condensed vector of the chosen file and sends it to cosine similarity scoring. '''

        idx_tf = self.documents[idx][1]
        return self._get_scores(idx_tf, idx)

    def get_document_by_index(self, idx):
        ''' Returns the original text of a document. '''
        return self.documents[idx][0]

def print_404():
    print(r"""

                       ._ o o
                       \_`-)|_
                    ,""       \ 
                  ,"  ## |   ಠ ಠ. 
                ," ##   ,-\__    `.
              ,"       /     `--._;)
            ,"     ## /
          ,"   ##    /

    404: No document matched your query
                    """)

def main():
    engine = IREngine()

    while(True):
        res = []
        user_input = input('Which search method would you like:\n '
                                '\t(1) Open query\n'
                                '\t(2) Similiar documents\n'
                                '\t(3) Union/Or open query\n'
                                '\t(4) Intersection open query\n'
                                '\t(q) Quit\n')

        if user_input == '1':
            query = input('Insert your search query (1-n terms): ')

            res = engine.process_query(query)

        elif user_input == '2':
            idx = int(input('Please enter the index of the document you would like to see more of: '))
            query = engine.get_document_by_index(idx)
            res = engine.more_like(idx)

        elif user_input == '3':
            query = my_tokenization_function(input('Insert your search query (1-n terms): '))
            res = engine.union(query)

        elif user_input == '4':
            query = my_tokenization_function(input('Insert 2 terms: '))
            res = engine.intersection(query[0], query[1])

        elif user_input == 'q':
            print('Goodbye.')
            break
        else:
            print('\n', 'Invalid input (q to exit).')
            print('-'*40)
            continue

        print('Your search query: {}'.format(query))
        print('-' * 40)
        print('your search results are: ')

        # Print query results
        if not res:
            print_404()
        elif user_input == '1' or user_input == '2':
            for doc in res:
                print('Document {}: {}'.format(doc[0], engine.get_document_by_index(doc[0])))
        elif user_input == '3' or user_input == '4':
            for idx in res:
                print('Document {}: {}'.format(idx, engine.get_document_by_index(idx)))

        print('')

if __name__ == '__main__':
    main()
