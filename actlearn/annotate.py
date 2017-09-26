"""command-line usage of the Annotator class.
"""

import argparse
import json
from actlearn.annotator import Annotator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help="Path to input file containing examples to be labeled. Must be a json file containing an array of dicts."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help="Path to json output file."
    )
    parser.add_argument(
        '-id',
        '--id_field',
        type=str,
        required=True,
        help="ID field."
    )
    parser.add_argument(
        '-id',
        '--id_field',
        nargs='+',
        required=False,
        default='all',
        help="Array of fields to display before each annotation."
    )
    parser.add_argument(
        '-s',
        '--autosave',
        action="store_true",
        default=True,
        help="Autosave every 50 annotations."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.input, 'r') as f:
        examples = json.load(f)
    annotator = Annotator(
        examples=examples,
        path=args.output,
        id_field=args.id_field,
        fields=args.fields,
        autosave=args.autosave
    )
    annotator.run()

if __name__ == '__main__': 
    main()
