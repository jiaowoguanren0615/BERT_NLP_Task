import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import transformers



class BERTDataset(Dataset):
    def __init__(self, review, args, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = transformers.BertTokenizer.from_pretrained('/usr/local/MyObjData/jupyterCode/CommonLit-Readability-Prize/bert-large-uncased', do_lower_case=True)
        self.max_len = args.max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())
        global inputs

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }


def build_dataset(args):
    data = pd.read_csv(args.data_root)
    data = data.sample(frac=1).reset_index(drop=True)

    # modify columns' names according to your own dataset
    data = data[['excerpt', 'target']]
    train_data, valid_data = train_test_split(data, train_size=.8, random_state=123)

    train_set = BERTDataset(
        review=train_data['excerpt'].values,
        args=args,
        target=train_data['target'].values
    )

    valid_set = BERTDataset(
        review=valid_data['excerpt'].values,
        args=args,
        target=valid_data['target'].values
    )
    return train_set, valid_set