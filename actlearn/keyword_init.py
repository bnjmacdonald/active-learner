"""provides keyword initialization for active learner."""

def keyword_init(documents, keywords, dictionary=None, threshold=0.01, find_keywords=False):
    """
    It must either be the case that (a) each doc in documents is a list of str
    representing each word and keywords is a list of str (thus dictionary can 
    be None); (b) each doc in documents is a list of int and keywords is a
    list of int; or (c) each doc in documents is a list of int, keywords is a
    list of str and dictonary is provided for mapping the keyword ints to str.

    Arguments:
        threshold (float): minimum proportion of words in document that are 
            keywords for it to be classified.
        find_keywords: if True, finds topn additional keywords that most
            commonly co-occur with the words in keywords and adds these to the
            keywords list.
    Returns:
        label: list of len(documents) elements where each element represents
            the assigned lable (0 or 1) of the document.
        counts: list of len(documents) elements where each element represents
            the number of words in the document that are in the keyword list.
        weights: list of len(documents) elements where each element represents
            the proportion of words in the document that are in the keyword
            list.
    """
    # Todo: add score to each word so that you can compute a weighted sum.
    doc = next(documents)
    if isinstance(doc[0], int) and isinstance(keywords[0], str):
        assert dictionary is not None, 'dictionary must be provided.'
        keywords_ints = []
        for kw in keywords:
            if kw in dictionary.token2id:
                keywords_ints.append(dictionary.token2id[kw])
            else:
                print('{0} not in dictionary'.format(kw))
        keywords = keywords_ints[:]
    assert len(keywords), 'keywords must contain at least one word.'
    assert type(doc[0]) == type(keywords[0]), 'type of each element in a document must match type of each keyword'
    counts = []
    weights = []
    labels = []
    # deals with first document
    keywords_in_doc = [w for w in doc if w in keywords]
    count = len(keywords_in_doc)
    doc_length = len(doc)
    weight = count / float(doc_length)
    label = int(weight > threshold)
    counts.append(count)
    weights.append(weight)
    labels.append(label)
    for i, doc in enumerate(documents):
        keywords_in_doc = [w for w in doc if w in keywords]
        count = len(keywords_in_doc)
        doc_length = len(doc)
        weight = count / float(doc_length)
        label = int(weight > threshold)
        counts.append(count)
        weights.append(weight)
        labels.append(label)
        if (i + 1) % 10000 == 0:
            print('processed {0} of {1} documents so far...'.format(i+1, len(documents)), end='\r')
    print('\n')
    return labels, counts, weights
