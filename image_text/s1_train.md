# Train
Tranining data uses Stanford Alpaca data. 
An example of using the data to train a very small model is below. 

## First run GPT model training

```
python3 dtwin/trainers/simple_trainer.py --model_name_or_path=facebook/opt-1.3b --data_path=stanford_alpaca/alpaca_data.json --output_dir=results/llama/trial --sample_size=1000 --limit_train_batches=1.0 --train_batch_size=1 --dev_batch_size=1 --num_train_epochs=1.0 --gradient_accumulation_steps=1 --n_layers=2 --dim=128
```

## Second run image-text alignment
```
python3 dtwin/trainers/simple_trainer.py --model_name_or_path=results/llama/trial --data_path=data/cc_sbu_align/filter_cap.json --vis_root=data/cc_sbu_align/image \
--output_dir=results/llama/trial_vis_text \
--sample_size=1000 --limit_train_batches=1.0 --train_batch_size=1 \
--per_device_train_batch_size=1 \
--dev_batch_size=1 --num_train_epochs=1.0 --gradient_accumulation_steps=1 --n_layers=2 --dim=128 \
--vit_depth=1 \
--vit_embed_dim=1408 \
--max_seq_len=34000 \
--run_name=small \
--task=image-text \
--dataloader_pin_memory=False \
--end_sym='\n'
```

