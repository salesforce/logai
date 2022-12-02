#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer

from logai.config_interfaces import Config


class TrainerConfig(Config):
    pretrain_from_scratch: bool = True
    model_name: str = 'bert-base-cased'
    model_dir: str = None
    mlm_probability: float = 0.15
    evaluation_strategy: str = 'steps'
    num_train_epochs: int = 20
    learning_rate: float = 1e-5
    logging_steps: int = 10
    per_device_train_batch_size: int = 50
    per_device_eval_batch_size: int = 256
    weight_decay: float = 0.0001
    save_steps: int = 500
    eval_steps: int = 500
    resume_from_checkpoint: bool = True
    trainer_config: dict = {
        'pretrain_from_scratch': True,
        'model_name': "bert-base-cased",
        'model_dir': '',
        'mlm_probability': 0.15,
        'evaluation_strategy': "steps",
        'num_train_epochs': 20,
        'learning_rate': 1e-5,
        'logging_steps': 10,
        'per_device_train_batch_size': 50,
        'per_device_eval_batch_size': 256,
        'weight_decay': 0.0001,
        'save_steps': 500,
        'eval_steps': 500,
        'resume_from_checkpoint': True,
    }

    def from_dict(self, config_dict):
        super().from_dict(config_dict)
        return


class Trainer:
    def __init__(self, config:TrainerConfig):
        self.config = config

    def train(self, train_dataset, valid_dataset):
        train_dataset, _, _ = load_dataset(train_file, self.data_column_name, self.special_tokens)
        valid_dataset, _, _ = load_dataset(valid_file, self.data_column_name, self.special_tokens)

        datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})

        tokenized_datasets = datasets.map(self.tokenize_function, batched=True, num_proc=4,
                                          remove_columns=[self.data_column_name])

        if os.path.exists(self.model_dir) and len(os.listdir(self.model_dir)) > 0:
            model_checkpoint = self.model_dir
            checkpoint_dir = 'checkpoint-' + str(max([int(x.split('-')[1]) for x in os.listdir(self.model_dir)]))
            model_checkpoint = os.path.abspath(os.path.join(model_checkpoint, checkpoint_dir))
        else:
            model_checkpoint = self.model_name

        if self.pretrain_from_scratch is False:
            model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        else:
            vocab_size = len(self.custom_vocab)
            config = BertConfig(vocab_size=vocab_size)
            model = BertForMaskedLM(config)
            model.tokenizer = self.tokenizer

        training_args = TrainingArguments(
            self.model_dir,
            evaluation_strategy=self.config.evaluation_strategy,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability,
                                                        pad_to_multiple_of=self.max_input_seq_len)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        return

    def _get_trainer(self):
        training_args = TrainingArguments(
            self.model_dir,
            evaluation_strategy=self.config.trainer_config['evaluation_strategy'],
            num_train_epochs=self.config.trainer_config['num_train_epochs'],
            learning_rate=self.config.trainer_config['learning_rate'],
            logging_steps=self.config.trainer_config['logging_steps'],
            per_device_train_batch_size=self.config.trainer_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config.trainer_config['per_device_eval_batch_size'],
            weight_decay=self.config.trainer_config['weight_decay'],
            save_steps=self.config.trainer_config['save_steps'],
            eval_steps=self.config.trainer_config['eval_steps'],
            resume_from_checkpoint=self.config.trainer_config['resume_from_checkpoint']
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator
        )
        return trainer

