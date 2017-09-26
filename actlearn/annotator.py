import os
import json
import random
from actlearn import utils
from collections import OrderedDict
# import numpy as np

class Annotator(object):
    """Implements a simple command-line annotator.
    
    Attributes:
        examples (list): list of dict containing data for each example to be
            labeled. Example::

                [{"_id": "0", "text": "blah blah blah"}, {"_id": "1", "text":
                "hello world"}, ...]

        path (str): string representing file path to json that will contain
            annotated data. The json file must contain a single dict with
            two keys: "meta" and "data". "meta" must be a dict with at
            least one key named "choices" which contains a dict of
            annotation choices. "data" is an array containing examples that
            have already been labeled. If no examples have been labeled,
            "data" must be an empty array.

            Example::

                {
                    "meta": {
                        "choices": {"0": "unemployment", "1": "crime", ...}
                        ...
                    },
                    "data": []
                }

        id_field: str representing the name of the field that uniquely
            identifies each example.
        fields: str or list of str. example fields to display for annotation.
            Either a list of string containing field names or "all", in which
            case all unique fields in the example are displayed.
        autosave: bool representing whether annotations should be auto-saved
            every 50 annotations. Default: True.

    Todo:
        * consider using mongoDB backend rather than relying on json structure.
    """
    prompt = 'LABEL ("q"=quit; "?"=skip): '

    def __init__(self, examples, path, id_field, fields='all', autosave=True):
        self.examples = examples
        self.shuffled_indices = list(range(len(self.examples)))
        random.shuffle(self.shuffled_indices)
        self.id_field = id_field
        if fields == 'all':
            # fields = sorted(set([d for d in examples for k in d.keys()]))
            fields = sorted(examples[0].keys())
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        self.path = path
        self.labeled_ids = []
        self.autosave = True
        self.autosave_every = 10
        
    def run(self):
        """runs the command-line interface for annotating examples."""
        self._load()
        assert len(self.labeled_ids) == len(self.labeled_examples)
        cmds = {'annotate': self._annotate, 'a': self._annotate, 'sample': self._sample, 's': self._sample, 'quit': self._quit, 'q': self._quit}
        prompt = 'what do you want to do ([a]nnotate/[s]ample/[q]uit)? '
        cmd = ''
        while cmd not in ['q', 'quit']:
            cmd = input(prompt).strip().lower()
            try:
                cmds[cmd]()
            except KeyError:
                print('"{0}" not recognized. Enter another value.'.format(cmd))
        return 0

    def _annotate(self):
        """requests user to annotate examples, one example at a time."""
        pop = True
        while len(self.shuffled_indices):
            if pop:
                example, ind = self._get_next_example()
            pop = True
            if example[self.id_field] in self.labeled_ids:
                continue
            self._display_example(example)
            self._display_choices()
            inp = input(self.prompt).strip().lower()
            if inp in ['q', 'quit']:
                self.shuffled_indices.insert(0, ind)
                break
            elif inp == '?':
                self.shuffled_indices.append(ind)
            elif inp in self.choices:
                self._process_annotation(inp, example)
            else:
                print('\n"{0}"" NOT RECOGNIZED. ENTER ANOTHER VALUE.\n'.format(inp))
                pop = False
        return 0

    def _get_next_example(self):
        """retrieves next example."""
        try:
            ind = self.shuffled_indices.pop(0)
            example = self.examples[ind]
        except IndexError:
            print('No examples left to annotate.')
            ind = None
            example = None
        return example, ind

    def _display_example(self, example):
        """displays example to user."""
        bar = '-' * 15
        print('\n\n{0}\nexamples labeled: {1}\nEXAMPLE:'.format(bar, len(self.labeled_examples)))
        print('ID: {0}'.format(example[self.id_field]))
        for field in self.fields:
            try:
                print('{0}: {1}'.format(field.upper(), example[field]))
            except KeyError:
                print('{0} not found.'.format(field.upper()))
        return 0

    def _process_annotation(self, label, example):
        """processes a user annotation."""
        assert 'label' not in example, 'example already has a label: {0}'.format(example)
        assert label in self.choices, '"{0}" not in label choices.'.format(label)
        example['label'] = self.choices[label]
        self.labeled_examples.append(example)
        self.labeled_ids.append(example[self.id_field])
        if self.autosave and len(self.labeled_examples) % self.autosave_every == 0:
            self._save()
        return 0

    def _display_choices(self):
        """displays choices to user."""
        print('\nCHOICES: ')
        for k, v in self.choices.items():
            print('{0:2s}: {1:30s}'.format(k, v))
        return 0

    def _quit(self):
        """quits the annotation interface."""
        inp = input('Save before quitting ([y]es/[n]o)? ').strip().lower()
        if inp in ['y', 'yes']:
            self._save()
        elif inp in ['n', 'no']:
            print('exited without saving.')
        else:
            print('"{0}" not recognized. Enter another value.'.format(inp))
            self._quit()
        return 0

    def _load(self):
        """loads previously annotated data."""
        with open(self.path, 'r') as f:
            json_data = json.load(f)
            self.labeled_examples = json_data['data']
            self.meta = json_data['meta']
            self.choices = OrderedDict(sorted([(str(choice), val) for choice, val in self.meta['choices'].items()], key=lambda x: x[0]))
        self.labeled_ids = []
        for ex in self.labeled_examples:
            self.labeled_ids.append(ex[self.id_field])
        assert len(self.choices)
        assert isinstance(self.labeled_examples, list)
        print('Imported {0} annotated samples.'.format(len(self.labeled_examples)))
        return 0

    def _save(self):
        """saves annotated data."""
        json_data = {'meta': self.meta, 'data': self.labeled_examples}
        with open(self.path, 'w') as f:
            json.dump(json_data, f, default=utils.datetime_objectid_handler)
        print('saved annotations to disk.')
        return 0

    def _sample(self, n_samples=5):
        """samples examples that have already been annotated."""
        n_samples = min(len(self.labeled_examples), n_samples)
        examples = random.sample(self.labeled_examples, n_samples)
        for example in examples:
            self._display_example(example)
            print('LABEL: {0}'.format(example['label']))
        return 0
