# **LLaMA 3.2 3B Fine-Tuning with LoRA and GGUF Conversion**

## **📌 Overview**
This project fine-tunes the **LLaMA 3.2 3B model** using **LoRA (Low-Rank Adaptation)** to create an efficient, customized AI model. The fine-tuned model is then **merged and converted to GGUF format** for deployment in **Ollama**.

## **🛠 Features**
✅ **Efficient LoRA Fine-Tuning** → Saves GPU memory and accelerates training.  
✅ **4-bit Quantization** → Enables fine-tuning on consumer GPUs (e.g., Tesla T4).  
✅ **Log Summarization** → AI-driven search and summarization of SMTP logs.  
✅ **Merge LoRA with Base Model** → No need to load adapters separately.  
✅ **GGUF Conversion for Ollama** → Fast and optimized model deployment.  

## **🚀 Installation**
### **1️⃣ Install Dependencies**
```bash
pip install -q pip3-autoremove
pip-autoremove torch torchvision torchaudio -y
pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install -q unsloth
pip install -q --upgrade --no-cache-dir transformers datasets
```

### **2️⃣ Download or Clone Repository**
```bash
git clone https://github.com/your-repo/llama-finetuning.git
cd llama-finetuning
```

## **📂 Project Structure**
```
├── training.py                # Fine-tuning script for LLaMA 3.2 3B with LoRA
├── merge_lora.py              # Merges LoRA fine-tuned adapters into the base model
├── convert_to_gguf.py         # Converts the merged model into GGUF format
├── requirements.txt           # List of dependencies
└── README.md                  # Documentation
```

## **🔹 Training Process**
### **1️⃣ Load and Preprocess the Dataset**
- Loads logs and structures them for fine-tuning.
- Supports JSON datasets with conversations.

### **2️⃣ Load Pretrained LLaMA Model**
- Uses **Unsloth** to efficiently load LLaMA 3.2 3B.
- Implements **4-bit quantization** for memory efficiency.

### **3️⃣ Apply LoRA for Fine-Tuning**
- Fine-tunes **only selected model layers** to reduce GPU load.
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
```

### **4️⃣ Train the Model**
- Uses **gradient accumulation** for better training efficiency.
```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
)
trainer.train()
```

### **5️⃣ Merge LoRA into the Base Model**
```python
model.load_adapter("/content/lora_model")
model.save_pretrained_merged("/content/merged_model", tokenizer, save_method="merged_16bit")
```

### **6️⃣ Convert Model to GGUF Format for Ollama**
```python
model.save_pretrained_gguf("/content/ollama_model", tokenizer)
```

## **🚀 Deployment with Ollama**
### **1️⃣ Transfer the GGUF Model to Your Server**
```bash
scp ollama_model.zip user@your-server-ip:/root/.ollama/models/
```

### **2️⃣ Register the Model in Ollama**
```bash
nano /root/.ollama/models/ollama_model/Modelfile
```
```
FROM /root/.ollama/models/ollama_model/unsloth.Q8_0.gguf
```
```bash
ollama create my_llama -f /root/.ollama/models/ollama_model/Modelfile
```

### **3️⃣ Run the Model in Ollama**
```bash
ollama run my_llama
```

## **📌 Future Improvements**
- **Integrate LangChain for better log retrieval.**
- **Use FAISS for fast similarity search in logs.**
- **Deploy as an API using FastAPI for real-time summarization.**


---

🎯 **Need help? Reach out via GitHub Issues!** 🚀

