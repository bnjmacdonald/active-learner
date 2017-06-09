"""implements ActiveLearner classes."""

import os
import json
import functools
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class ActiveLearner(object):
    """Abstract implementation of an active learning system.
    
    User satisfaction will depend a lot on intelligent initialization and
    sampling such that learning happens very quickly.

    This approach is attractive because it allows for a model-based
    user-driven information extraction system that anybody can train. It is
    also subject to manipulation though.

    Attributes:
        documents (list): list of str containing the raw text of each
            document.
        ids (list): list of int containing unique identifiers of each
            document.
        out_path (str): string representing output directory.
        labels_dict (dict): dictionary of int->str mappings representing the
            integer key associated with each label (e.g. {0: 'unemployment'}).

    Todo:
        * labels_dict cannot be None for BinaryLearner
    """
    # out_fname = 'labels.csv'
    labels_dict_filename = 'labels_dict.json'
    prompt = 'LABEL ("q" to quit): '
    
    model = None

    def __init__(self, documents, ids, out_path, X, labels=None, labels_dict=None, update_every=10, out_fname='labels.csv', verbose=0):
        self.documents = documents
        self.X = X
        self.ids = ids
        self.out_path = out_path
        self.update_every = update_every
        self.verbose = verbose
        self.out_fname = out_fname
        if labels is None:
            labels = np.full((len(documents),), np.nan)
        if labels_dict is None:
            try:
                labels_dict = self.read_labels_dict()
            except:
                labels_dict = {}
        try:
            labels_subset, labeled_rows = self.read_labels()
        except:
            labels_subset, labeled_rows = [], []
        labels = labels.astype(np.float64)
        labels[labeled_rows] = labels_subset
        self.labels_dict = labels_dict
        self.labels = labels
        self.labeled_rows = labeled_rows
        self.class_preds = None
        self.class_probs = None
        self.sampling_probs = None

    def run(self):
        cmds = {'train': self.update, 't': self.update, 'annotate': self.annotate, 'a': self.annotate, 'sample': self.sample, 's': self.sample}
        prompt = 'what do you want to do ([t]rain/[a]nnotate/[s]ample/[q]uit)? '
        cmd = ''
        while cmd not in ['q', 'quit']:
            cmd = input(prompt)
            while cmd not in cmds.keys():
                if cmd in ['q', 'quit']:
                    return 0
                cmd = input(prompt)
            cmds[cmd]()
        return 0

    def update(self, manual_only=True):
        if manual_only:
            if not len(self.labeled_rows) > 1:
                print("you can't train the model if you haven't annotated any documents! Either run self.initialize() or manually annotate at least two documents first.")
                return
            X = self.X[self.labeled_rows]
            y = np.array(self.labels)[self.labeled_rows]
        else:
            X = self.X
            y = self.labels
        assert X.shape[0] == y.shape[0]
        y2 = y[~np.isnan(y)].astype(np.int64)
        X2 = X[~np.isnan(y)]
        self.train(X2, y2)
        self.predict(self.X)
        if self.class_preds is not None:
            if manual_only:
                preds = self.class_preds[self.labeled_rows]
                preds = preds[~np.isnan(y)]
            else:
                preds = self.class_preds[~np.isnan(self.labels)]
            if self.verbose:
                print('Performance:', self.evaluate(y2, preds))

    def train(self, X, labels):
        if self.verbose:
            print('training model...')
        # y2 = labels[~np.isnan(labels)].astype(np.int64)
        # X2 = X[~np.isnan(labels)]
        try:
            self.model.fit(X, labels)
        except ValueError as e:
            print('Could not train.')
            print(e)
        except IndexError as e:
            print('Could not predict.')
            print(e)

    def predict(self, X):
        if self.verbose:
            print('predicting classes...')
        # cross_val_score(self.model, X, y, cv=5, scoring='f1_macro')
        try:
            self.class_preds = self.model.predict(X)
            self.class_probs = self.model.predict_proba(X)
            if self.verbose:
                print('Predictions:', np.bincount(self.class_preds))
                # print(np.bincount(self.class_preds))
        except sklearn.exceptions.NotFittedError as e:
            print('Could not predict.')
            print(e)
        except IndexError as e:
            print('Could not predict.')
            print(e)

    def evaluate(self, labels, preds):
        assert labels.shape[0] == preds.shape[0]
        f1 = metrics.f1_score(labels, preds, average='macro')
        recall = metrics.recall_score(labels, preds, average='macro')
        precision = metrics.precision_score(labels, preds, average='macro')
        accuracy = metrics.accuracy_score(labels, preds)
        confusion = metrics.confusion_matrix(labels, preds)
        perf = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'confusion': confusion.tolist(),
            'n_obs': labels.shape
            # 'roc': roc
        }
        return perf

    def annotate(self):
        self.update_sampling_probs()
        label = ''
        self.print_help_msg()
        # f = open(os.path.join(self.out_path, self.annotations_filename), 'a')
        # f_labels = open(os.path.join(self.out_path, self.out_fname), 'a')
        # writer = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=',')
        while label != 'q':
            if len(set(self.labeled_rows)) == len(self.documents):
                print('You have annotated all examples in the corpus.')
                break
            sample = np.random.choice(np.arange(len(self.documents)), replace=False, p=self.sampling_probs)
            while sample in self.labeled_rows:
                sample = np.random.choice(np.arange(len(self.documents)), replace=False, p=self.sampling_probs)
            # requests label from user
            self.display_sample(sample)
            label = input(self.prompt)
            label = label.strip()
            if label == 'q':
                break
            self.process_annotation(label, sample)
            if len(self.labeled_rows) % 10 == 0:
                self.save_labels()
            if len(self.labeled_rows) % self.update_every == 0:
                self.update(manual_only=True)
                self.update_sampling_probs()
            # sampled.append(self.ids[sample])
    
    def fix_label(self, uid, label):
        if label not in self.labels_dict:
            raise ValueError('label not in self.labels_dict')
        row = np.where(np.array(self.ids) == uid)[0][0]
        orig_label = self.labels[row]
        self.labels[row] = label
        print('Replaced label {0} with {1} for document {2}'.format(orig_label, label, uid))
        if row not in self.labeled_rows:
            self.labeled_rows.append(row)
            
    def display_sample(self, i):
        pk = self.ids[i]
        bar = '-'*25
        print('\n\n{0}\nDocuments labeled: {1}\nDocument ID: {2}\nPredicted values:'.format(bar, len(self.labeled_rows), pk))
        for k, v in self.labels_dict.items():
            try:
                prob = round(self.class_probs[i][k], 3)
            except:
                prob = '?'
            print('p({0:d}) = p({1}) = {2}'.format(k, v, prob))
        print(self.documents[i])

    def print_help_msg(self):
        bar = '='*30
        help_msg = '\n{0}\nPress "q" to quit. Separate each label by a comma (",").\nLabels:'.format(bar)
        print(help_msg)
        if not len(self.labels_dict):
            print('\tno labels yet...')
        for k, v in self.labels_dict.items():
            print('\t{0:3d}: {1:30s}'.format(k, v))

    def print_err_msg(self, inp):
        print('"{0}" exceeds max length or not in acceptable labels or already has a numeric key'.format(inp))

    def read_labels(self):
        data = pd.read_csv(os.path.join(self.out_path, self.out_fname))
        data['id'] = data['id'].astype(str)
        labels, labeled_ids = data['label'].tolist(), data['id'].tolist()
        assert type(labeled_ids[0]) != type(self.ids[0])
        labeled_rows = []
        for i in labeled_ids:
            try:
                ix = self.ids.index(i)
                labeled_rows.append(ix)
            except ValueError:
                print('WARNING: labeled ID {0} not in self.ids. This ID will be deleted on call to self.save_labels.'.format(i))
        if self.verbose:
            print('imported {0} annotated samples.'.format(len(labeled_rows)))
        return labels, labeled_rows

    def save_labels(self):
        labeled_ids = [self.ids[i] for i in self.labeled_rows]
        labeled_subset = [self.labels[i] for i in self.labeled_rows]
        data = pd.DataFrame.from_dict({'id': labeled_ids, 'label': labeled_subset}, orient='columns')
        data.to_csv(os.path.join(self.out_path, self.out_fname), index=False)
        if self.verbose:
            print('saved labels to disk.')

    def save_preds(self, fname):
        if self.class_preds is None:
            raise IOError('class_preds is None. Run self.predict() first.')
        data = pd.Series(self.class_preds, index=self.ids)
        data.to_csv(os.path.join(self.out_path, fname), index=True)

    def read_labels_dict(self):
        with open(os.path.join(self.out_path, self.labels_dict_filename), 'r') as f:
            labels_dict = json.load(f)
            # labels_dict = {}
            # for l in f.readlines():
            #     key, label = l.strip().split(':')
            #     labels_dict[int(key)] = label
        if self.verbose:
            print('imported labels dictionary:\n{0}'.format(labels_dict))
        return labels_dict

    def update_sampling_probs(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def process_annotation(self):
        raise NotImplementedError

    def sample(self, n_samples):
        raise NotImplementedError

class MultilabelActiveLearner(ActiveLearner):
    """Implements a multi-label active learning system.
    
    Algorithm:
        initialize set of document classifications.
        while user doesn't quit:
            sample a document
            ask user for label
            process label and save to disk
        re-train prediction function on all data points (or on next batch).
        re-estimate prediction function on entire dataset.
    
    Notes:
        two initialization use cases:
            (1) there are zero annotations:
                labels_dict = None
                annotations = empty pd.DataFrame
                class_probs = None
                sampling_probs = 1.0/len(self.document)
            (2) there are pre-existing annotations:
                labels_dict = dict from file
                annotations = pd.DataFrame from file
                class_probs = fit model
                sampling_probs = compute based on fit.
    Usage::
        
        learner = MultilabelActiveLearner(documents, ids, out_path, labels_dict, verbose)
        learner.annotate()
    
    Todo:
        * update_sampling_probs is not well thought out for the multilabel problem.
    """
    C = np.random.choice(np.logspace(-4, 4, num=5000, base=10), size=10, replace=False)
    cv = 5
    # model = LogisticRegressionCV(penalty='l2', Cs=C, cv=cv, scoring='f1_macro', multi_class='multinomial', n_jobs=4, refit=True, verbose=1)
    estimator = LogisticRegression(penalty='l2', multi_class='multinomial', n_jobs=4, solver='lbfgs')
    param_grid = {'C': C}
    model = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='f1_macro', n_jobs=4, cv=cv, verbose=1)
    max_length = 300

    def process_annotation(self, label, label_row):
        """processes a user annotation, saving annotation to file."""
        while len(label) > self.max_length or len(label) == 0:
            print(self.print_err_msg(label))
            label = input(self.prompt)
            label = label.strip()
            if label == 'q':
                return
        if label == '?':
            label = np.nan
            self.labels[label_row] = label
            self.labeled_rows.append(label_row)
        else:
            labels = label.split(',')
            for label in labels:
                label = label.strip()
                if not len(label): continue
                if label in self.labels_dict.values():
                    label = [str(k) for k, v in self.labels_dict.items() if v == label][0]
                if not label.isdigit():
                    label = self.update_labels_dict(label)
                label = int(label)
                self.labels[label_row] = label
                self.labeled_rows.append(label_row)
                # f.write('{0},{1}\n'.format(self.ids[sample], label))

    def update_sampling_probs(self):
        if self.class_probs is None:
            pred_uncertainty = np.full((len(self.documents),), 1.0)
        else:
            pred_uncertainty = np.sum(np.multiply(self.class_probs, self.class_probs), axis=1)
        sampling_probs = np.divide(pred_uncertainty, pred_uncertainty.sum())
        self.sampling_probs = sampling_probs

    def update_labels_dict(self, label):
        """adds a new key-value pair to self.labels_dict and returns key."""
        new_label_key = max(self.labels_dict.keys()) + 1 if len(self.labels_dict) else 0
        self.labels_dict[new_label_key] = label
        print('added new label: {0}={1}'.format(new_label_key, label))
        # f_labels.write('{0}:{1}\n'.format(new_label_key, label))
        return new_label_key

    def sample(self, n_samples=1):
        # rows = np.where(self.class_preds == 1)[0]
        # np.array(self.documents)[self.class_preds == 1]
        sample = np.random.choice(range(len(self.documents)), size=n_samples, replace=False)
        # sampled_docs = [self.documents[i] for i in rows]
        for i in sample:
            self.display_sample(i)
            print('Predicted class: ', self.labels_dict[self.class_preds[i]])



class BinaryActiveLearner(ActiveLearner):
    """Implements a two class active learning system.

    Algorithm:
        initialize set of documents classified as "1".
        initialize prediction function 
        while user doesn't quit:
            sample a document
            ask user for label
            process label and save to disk
        re-train prediction function on all data points (or on next batch).
        re-estimate prediction function on entire dataset.
    """

    acceptable_labels = ['0', '1', '?']
    C = np.random.choice(np.logspace(-4, 4, num=5000, base=10), size=10, replace=False)
    cv = 5
    model = LogisticRegressionCV(penalty='l2', Cs=C, cv=cv, scoring='f1_macro', n_jobs=4, refit=True, verbose=1)

    def __init__(self, *args, **kwargs):
        # self.out_fname = self.out_fname.replace('.csv', '_{0}.csv'.format(issue))
        # self.issue = issue
        super().__init__(*args, **kwargs)
        
        
    def update_sampling_probs(self):
        if self.class_probs is None:
            pred_uncertainty = np.full((len(self.documents),), 1.0)
        else:
            pred_uncertainty = np.prod(self.class_probs, axis=1)
        sampling_probs = np.divide(pred_uncertainty, pred_uncertainty.sum())
        self.sampling_probs = sampling_probs

    def process_annotation(self, label, label_row):
        """processes a user annotation, saving annotation to file."""
        while label not in self.acceptable_labels or len(label) == 0:
            print(self.print_err_msg(label))
            label = input(self.prompt)
            label = label.strip()
            if label == 'q':
                break
        if label == 'q':
            return
        if label == '?':
            label = np.nan
        self.labels[label_row] = label
        self.labeled_rows.append(label_row)
        # f.write('{0},{1}\n'.format(self.ids[sample], label))

    def sample(self, n_samples=5, classes=None):
        if isinstance(classes, int):
            classes = [classes]
        if classes is None:
            classes = list(self.labels_dict.keys())
        # rows = np.where(self.class_preds == 1)[0]
        rows = np.array([i for i, pred in enumerate(self.class_preds) if pred in classes])
        
        # np.array(self.documents)[self.class_preds == 1]
        sample = np.random.choice(rows, size=n_samples, replace=False)
        # sampled_docs = [self.documents[i] for i in rows]
        for i in sample:
            self.display_sample(i)
            print('Predicted class: ', self.labels_dict[self.class_preds[i]])
            # print('-'*30)
            # print('Document: ', self.ids[i])
            # print('Predicted label: ', self.labels_dict[self.class_preds[i]])
            # print('Predictions: ', self.class_probs[i])
            # print(self.documents[i])

class MultiDimensionalLearner(object):
    """Implements a multi-dimensional learner.

    Todo:
        * replicating documents, ids, and X for each learner is really memory-
            inefficient. Alter this so that only one copy of each is needed.
        * implement sampling strategies.
    """

    prompt = 'LABEL for learner {0} ("q" to quit): '

    def __init__(self, n_learners, documents, ids, out_path, X, learner_type='binary', labels_dicts=None, update_every=10, out_fname='labels.csv', verbose=0):
        if labels_dicts is not None and n_learners != len(labels_dicts):
            raise RuntimeError('n_learners must be the same as the number of elements in labels_dicts.')
        if learner_type == 'binary':
            learner_cls = BinaryActiveLearner
        elif learner_type == 'multilabel':
            learner_cls = MultilabelActiveLearner
        else:
            raise NotImplementedError
        learner_inits = []
        for i in range(n_learners):
            labels_dict = labels_dicts[i] if labels_dicts is not None else None
            this_out_fname = out_fname.replace('.csv', '_md{0}.csv'.format(i))
            learner = learner_cls(documents=documents, ids=ids, out_path=out_path, X=X, labels=None, labels_dict=labels_dict, update_every=update_every, out_fname=this_out_fname, verbose=verbose)
            learner_inits.append(learner)
        self.learners = learner_inits
        self.documents = documents
        self.X = X
        self.ids = ids
        self.out_path = out_path
        self.update_every = update_every
        self.verbose = verbose

    def run(self):
        cmds = {'train': self.update, 't': self.update, 'annotate': self.annotate, 'a': self.annotate, 'sample': self.sample, 's': self.sample}
        prompt = 'what do you want to do ([t]rain/[a]nnotate/[s]ample/[q]uit)? '
        cmd = ''
        while cmd not in ['q', 'quit']:
            cmd = input(prompt)
            while cmd not in cmds.keys():
                if cmd in ['q', 'quit']:
                    return 0
                cmd = input(prompt)
            cmds[cmd]()
        return 0

    def update(self, manual_only=True):
        for i, learner in enumerate(self.learners):
            if manual_only:
                if not len(learner.labeled_rows) > 1:
                    print("you can't train the model if you haven't annotated any documents! Either run learner.initialize() or manually annotate at least two documents first.")
                    return
                X = learner.X[learner.labeled_rows]
                y = np.array(learner.labels)[learner.labeled_rows]
            else:
                X = learner.X
                y = learner.labels
            assert X.shape[0] == y.shape[0]
            y2 = y[~np.isnan(y)].astype(np.int64)
            X2 = X[~np.isnan(y)]
            learner.train(X2, y2)
            learner.predict(learner.X)
            if manual_only:
                preds = learner.class_preds[learner.labeled_rows]
                preds = preds[~np.isnan(y)]
            else:
                preds = learner.class_preds[~np.isnan(learner.labels)]
            if learner.verbose:
                print('Learner {0} performance:'.format(i), learner.evaluate(y2, preds))

    def annotate(self):
        for i, learner in enumerate(self.learners):
            learner.update_sampling_probs()
        self.learners[0].print_help_msg()
        sampling_probs = functools.reduce(np.multiply, [l.sampling_probs for l in self.learners])
        sampling_probs = sampling_probs / sampling_probs.sum()
        sample = np.random.choice(np.arange(len(self.documents)), replace=False, p=sampling_probs)
        while sample in self.learners[-1].labeled_rows:
            sample = np.random.choice(np.arange(len(self.documents)), replace=False, p=sampling_probs)
        if len(set(self.learners[-1].labeled_rows)) == len(learner.documents):
            print('You have annotated all examples in the corpus.')
            return
        self.display_sample(sample)
        for i, learner in enumerate(self.learners):
            # requests label from user
            label = ''
            label = input(self.prompt.format(i))
            label = label.strip()
            if label == 'q':
                break
            learner.process_annotation(label, sample)
            if len(learner.labeled_rows) % 10 == 0:
                learner.save_labels()
            # sampled.append(learner.ids[sample])
        if len(self.learners[-1].labeled_rows) % self.update_every == 0:
            self.update(manual_only=True)
            # for i, learner in enumerate(self.learners):
            #     learner.update_sampling_probs()

    def display_sample(self, i):
        pk = self.ids[i]
        bar = '-'*25
        print('\n\n{0}\nDocuments labeled: {1}\nDocument ID: {2}'.format(bar, len(self.learners[-1].labeled_rows), pk))
        for j, learner in enumerate(self.learners):
            print('\nPredicted values for learner {0}:'.format(j))
            for k, v in learner.labels_dict.items():
                try:
                    prob = round(learner.class_probs[i][k], 3)
                except:
                    prob = '?'
                print('\tp({0:d}) = p({1}) = {2}'.format(k, v, prob))
        print(self.documents[i])

    def save_labels(self):
        raise NotImplementedError

    def save_preds(self):
        raise NotImplementedError
