import os
import argparse
import sys
sys.path.append('/home/sy/project/albert_srl/')
from pprint import pprint

from srl.module import SRL

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("Base path : {}".format(path))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(path, 'model/pytorch_model.bin'),
        type=str,
        required=False,
        help="The path of model!",
    )
    parser.add_argument(
        '--SPECIAL_TOKEN',
        default={"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"},
        type=dict,
        required=False,
        help='The dictionary of special tokens!'
    )
    parser.add_argument(
        '--LABEL2I',
        default={
                 '[PAD]': 0,
                 '[UNK]': 1,
                 '[SEP]': 2,
                 '[CLS]': 3,
                 'O': 4,
                 'S-REL': 5,
                 'S-ARG3': 6,
                 'S-ARG2': 7,
                 'S-ARGM-LOC': 8,
                 'S-ARGM-TPC': 9,
                 'S-ARGM-CND': 10,
                 'S-ARGM-EXT': 11,
                 'S-ARGM-MNR': 12,
                 'S-ARGM-ADV': 13,
                 'S-ARG1': 14,
                 'S-ARGM-DIS': 15,
                 'S-ARG0': 16,
                 'S-ARGM-TMP': 17,
                 'B-ARG1': 18,
                 'I-ARG1': 19,
                 'E-ARG1': 20,
                 'B-REL': 21,
                 'I-REL': 22,
                 'E-REL': 23,
                 'B-ARG0': 24,
                 'I-ARG0': 25,
                 'E-ARG0': 26,
                 'B-ARGM-ADV': 27,
                 'I-ARGM-ADV': 28,
                 'E-ARGM-ADV': 29,
                 'B-ARGM-TMP': 30,
                 'I-ARGM-TMP': 31,
                 'E-ARGM-TMP': 32,
                 'B-ARG2': 33,
                 'I-ARG2': 34,
                 'E-ARG2': 35,
                 'B-ARGM-LOC': 36,
                 'I-ARGM-LOC': 37,
                 'E-ARGM-LOC': 38,
                 'B-ARGM-MNR': 39,
                 'I-ARGM-MNR': 40,
                 'E-ARGM-MNR': 41,
                 'B-ARGM-CND': 42,
                 'I-ARGM-CND': 43,
                 'E-ARGM-CND': 44,
                 'B-ARGM-TPC': 45,
                 'I-ARGM-TPC': 46,
                 'E-ARGM-TPC': 47,
                 'B-ARGM-BNF': 48,
                 'I-ARGM-BNF': 49,
                 'E-ARGM-BNF': 50,
                 'B-ARGM-PRP': 51,
                 'I-ARGM-PRP': 52,
                 'E-ARGM-PRP': 53,
                 'B-ARGM-FRQ': 54,
                 'E-ARGM-FRQ': 55,
                 'B-ARGM-DIR': 56,
                 'I-ARGM-DIR': 57,
                 'E-ARGM-DIR': 58,
                 'B-ARGM-EXT': 59,
                 'I-ARGM-EXT': 60,
                 'E-ARGM-EXT': 61,
                 'B-ARGM-DIS': 62,
                 'I-ARGM-DIS': 63,
                 'E-ARGM-DIS': 64,
                 'B-ARG3': 65,
                 'E-ARG3': 66,
                 'I-ARG3': 67,
                 'B-ARG4': 68,
                 'I-ARG4': 69,
                 'E-ARG4': 70
                 },
        type=dict,
        required=False,
        help='The dictionary of label2i!'
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join(path, 'data/srl/example.train'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument(
        '--dev_path',
        default=os.path.join(path, 'data/srl/example.dev'),
        type=str,
        required=False,
        help='The path of dev set!'
    )
    parser.add_argument(
        '--test_path',
        default=None,
        type=str,
        required=False,
        help='The path of test set!'
    )
    parser.add_argument(
        '--log_path',
        default=None,
        type=str,
        required=False,
        help='The path of Log!'
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=32, type=int, required=False, help="Batch size!"
    )
    parser.add_argument('--step_size', default=50, type=int, required=False, help='lr_scheduler step size!')
    parser.add_argument("--lr", default=0.0001, type=float, required=False, help="Learning rate!")
    parser.add_argument('--clip', default=5, type=float, required=False, help='Clip!')
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=300, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument('--train', default='flase', type=str, required=False, help='Train or predict!')
    args = parser.parse_args()
    train_bool = lambda x:x.lower() == 'true'
    srl = SRL(args)
    if train_bool(args.train):
        srl.train()
    else:
        srl.load()
        # ner.test(args.test_path)
        text = '代表朝方对中国党政领导人和人民哀悼金日成主席逝世表示深切谢意'
        predicates_indicator = [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,1,1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]
        assert len(text) == len(predicates_indicator)
        pprint(srl.predict(text, predicates_indicator))


