import transformers
import torch.nn as nn



class BERT_BASE_UNCASED(nn.Module):
    def __init__(self, bert_path, num_classes, **kwargs):
        super(BERT_BASE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.drop = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(768, eps=1e-5)
        self._init_weights(self.layer_norm)
        self.fc = nn.Linear(768, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.fc(output)
        return output


class BERT_LARGE_UNCASED(nn.Module):
    def __init__(self, bert_path, num_classes, **kwargs):
        super(BERT_LARGE_UNCASED, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-5)
        self._init_weights(self.layer_norm)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        # print(_.shape)
        # print(output.shape)
        output = self.drop(output)
        output = self.fc(output)
        return output


class DBERT_BASE_UNCASED(nn.Module):
    def __init__(self, bert_path, num_classes, **kwargs):
        super(DBERT_BASE_UNCASED, self).__init__()
        self.dbert = transformers.DistilBertModel.from_pretrained(bert_path)
        self.layer_norm = nn.LayerNorm(768, eps=1e-5)
        self._init_weights(self.layer_norm)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(768, num_classes)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask):
        output = self.dbert(ids, attention_mask=mask, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output


def build_model(args):
    assert args.model.upper() in ['BERT_BASE_UNCASED', 'BERT_LARGE_UNCASED', 'DBERT_BASE_UNCASED'], \
        'You must choose one of these models'

    if args.model.upper == 'BERT_BASE_UNCASED':
        model = BERT_BASE_UNCASED(args.bert_path, args.nb_classes)
    elif args.model.upper == 'BERT_LARGE_UNCASED':
        model = BERT_LARGE_UNCASED(args.bert_path, args.nb_classes)
    elif args.model.upper == 'DBERT_BASE_UNCASED':
        model = BERT_BASE_UNCASED(args.bert_path, args.nb_classes)
    else:
        model = BERT_LARGE_UNCASED(args.bert_path, args.nb_classes)
    return model