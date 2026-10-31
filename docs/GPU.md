# CPU / GPU

Everything runs on CPU (`device: cpu`). Training BERT or ResNet backbones on the full MVSA data is slow on CPU;
use an NVIDIA GPU (>= 8 GB VRAM recommended for BERT-base with batch 16) and install a CUDA build of PyTorch from
https://pytorch.org/get-started/locally/. Then `--device auto` (default) or `--device cuda` uses it; set
`training.use_amp: true` for mixed precision. BERT weights and tokenizer are downloaded from the Hugging Face hub on first use
(internet needed once; they are then cached), or point `data.bert_tokenizer` and `models.*.pretrained_name` at a local folder.
