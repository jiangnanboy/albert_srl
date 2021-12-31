import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertTokenizer, AlbertModel, AlbertConfig
import sys

sys.path.append('/home/sy/project/albert_srl/')

from utils.log import logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(model_path, special_token):
    logger.info('loading tokenizer {}'.format(model_path))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def load_config(pretrained_model_path, tokenizer):
    albertConfig = AlbertConfig.from_pretrained(pretrained_model_path,
                                                cls_token_id=tokenizer.cls_token_id,
                                                sep_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                unk_token_id=tokenizer.unk_token_id,
                                                output_attentions=False,  # whether or not return [attentions weights]
                                                output_hidden_states=False)  # whether or not return [hidden states]
    return albertConfig

def load_pretrained_model(pretrained_model_path, tokenizer, special_token):
    logger.info('loading pretrained model {}'.format(pretrained_model_path))
    albertConfig = load_config(pretrained_model_path, tokenizer)
    model = AlbertModel.from_pretrained(pretrained_model_path, config=albertConfig)

    if special_token:
        # resize special token
        model.resize_token_embeddings(len(tokenizer))

    return model, albertConfig

def build_model(albertConfig):
    logger.info('build albertmodel!')
    model = AlbertModel(config=albertConfig)
    return model

class AlbertCrf(nn.Module):
    def __init__(self, config, pretrained_model, tag_num):
        super(AlbertCrf, self).__init__()
        self.config = config
        self.model = pretrained_model
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size + 1, tag_num) # add predicates indicator
        self.crf = CRF(num_tags=tag_num, batch_first=True)

    def loss(self, input_idx, token_type_ids=None, attention_mask=None, predicates=None, labels=None, label_mask=None):
        outputs = self.model(input_ids=input_idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = torch.cat((sequence_output, predicates.unsqueeze(-1)), -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # outputs = (logits,)s
        loss = self.crf(emissions=logits, tags=labels, mask=label_mask.byte())
        # outputs = (-1 * loss,) + outputs
        # return outputs # (loss), scores
        return loss

    def forward(self, input_idx, token_type_ids=None, attention_mask=None, predicates=None, label_mask=None):
        outputs = self.model(input_ids=input_idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = torch.cat((sequence_output, predicates.unsqueeze(-1)), -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        tags = self.crf.decode(emissions=logits, mask=label_mask.byte())
        return tags

