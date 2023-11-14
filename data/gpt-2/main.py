from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def fine_tune_gpt2(model_name, train_file, output_dir):
    custom_config = GPT2Config(
        n_embd=768,  # Número de dimensões do espaço de embedding
        n_head=12,  # Número de cabeças de atenção
        n_layer=12,  # Número de camadas do modelo
    )

    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    n_head = model.config.n_head
    n_layer = model.config.n_layer

    # Imprima os valores
    # print(f"Número de cabeças (n_head): {n_head}")
    # print(f"Número de camadas (n_layer): {n_layer}")
    print(f"Número de camadas (batch_size): {model.config}")

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_file, block_size=128
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Set training arguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


fine_tune_gpt2("gpt2", "intents/mental_health_data.txt", "output")
