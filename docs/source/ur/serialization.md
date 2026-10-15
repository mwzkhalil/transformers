<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# پروڈکشن کے لیے برآمد کرنا

بہتر رن ٹائمز اور آلات کے لیے Transformers' ماڈلز کو مختلف فارمیٹس میں برآمد کریں۔ وہی ماڈل کلاؤڈ فراہم کنندگان پر تعینات کریں یا اسے موبائل اور ایج ڈیوائسز پر چلائیں۔ آپ کو ہر تعیناتی ماحول کے لیے ماڈل کو شروع سے دوبارہ لکھنے کی ضرورت نہیں ہے۔ کسی بھی انفرنس ماحولیاتی نظام میں آزادانہ طور پر تعینات کریں۔

## ExecuTorch

[ExecuTorch](https://pytorch.org/executorch/stable/index.html) موبائل اور ایج ڈیوائسز پر PyTorch ماڈلز چلتا ہے۔ یہ ایک ماڈل کو معیاری آپریٹرز کے گراف میں برآمد کرتا ہے، گراف کو ایک ExecuTorch پروگرام میں مرتب کرتا ہے، اور اسے ٹارگٹ ڈیوائس پر چلاتا ہے۔ رن ٹائم ہلکا ہے اور وقت سے پہلے عملدرآمد کے منصوبے کا حساب لگاتا ہے۔

ماخذ سے [Optimum ExecuTorch](https://huggingface.co/docs/optimum-executorch/en/index) انسٹال کریں۔

```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
```

CLI ٹول کے ساتھ ExecuTorch میں Transformers ماڈل ایکسپورٹ کریں۔

```bash
optimum-cli export executorch \
    --model "Qwen/Qwen3-8B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```

تمام برآمدی اختیارات کو دیکھنے کے لیے درج ذیل کمانڈ کو چلائیں۔

```bash
optimum-cli export executorch --help
```

## او این این ایکس

[ONNX](http://onnx.ai) مختلف فریم ورکس سے ماڈلز بیان کرنے کے لیے ایک مشترکہ زبان ہے۔ یہ ماڈلز کو معیاری آپریٹرز کے گراف کے طور پر اچھی طرح سے متعین اقسام، شکلوں اور میٹا ڈیٹا کے ساتھ پیش کرتا ہے۔ ماڈلز کومپیکٹ پروٹوبف فائلوں میں سیریلائز کریں جنہیں آپ بہترین رن ٹائمز اور انجنوں میں تعینات کر سکتے ہیں۔

[کنفیگریشن آبجیکٹ کے ساتھ ONNX کو بہترین ONNX](https://huggingface.co/docs/optimum-onnx/index) ماڈلز برآمد کرتا ہے۔ یہ بہت سے [architectures](https://huggingface.co/docs/optimum-onnx/onnx/overview) کو سپورٹ کرتا ہے اور آسانی سے قابل توسیع ہے۔ CLI ٹول کے ذریعے یا پروگرام کے ذریعے ماڈلز برآمد کریں۔

انسٹال کریں [Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index).

```bash
uv pip install optimum-onnx
```

### optimum-cli

ایکسپورٹ کرنے کے لیے ایک ماڈل اور `--model` دلیل کے ساتھ آؤٹ پٹ ڈائرکٹری کی وضاحت کریں۔

```bash
optimum-cli export onnx --model Qwen/Qwen3-8B Qwen/Qwen3-8b-onnx/
```

تمام دستیاب دلائل دیکھنے کے لیے درج ذیل کمانڈ کو چلائیں یا مزید تفصیلات کے لیے [OnNX پر ماڈل ایکسپورٹ کریں optimum.exporters.onnx](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model) گائیڈ سے رجوع کریں۔

```bash
optimum-cli export onnx --help
```

مقامی ماڈل برآمد کرنے کے لیے، اسی ڈائرکٹری میں وزن اور ٹوکنائزر فائلوں کو محفوظ کریں۔ ڈائرکٹری کا راستہ `--model` آرگیومینٹ تک پہنچائیں اور `--task` دلیل کا استعمال کریں تاکہ [task](https://huggingface.co/docs/optimum/exporters/task_manager#transformers) کی وضاحت کریں۔ اگر آپ `--task` فراہم نہیں کرتے ہیں، تو سسٹم اسے ماڈل سے خود بخود اندازہ لگاتا ہے یا ٹاسک مخصوص ہیڈ کے بغیر فن تعمیر کا استعمال کرتا ہے۔

```bash
optimum-cli export onnx --model path/to/local/model --task text-generation Qwen/Qwen3-8b-onnx/
```

ماڈل کو کسی بھی [رن ٹائم](https://onnx.ai/supported-tools.html#deployModel) کے ساتھ تعینات کریں جو ONNX کو سپورٹ کرتا ہو، بشمول ONNX رن ٹائم۔

```py
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b-onnx")
model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b-onnx")
inputs = tokenizer("Plants generate energy through a process known as ", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs))
```

### optimum.onnxruntime

Transformers' ماڈلز پروگرام کے لحاظ سے بہترین ONNX کے ساتھ برآمد کریں۔ [`~optimum.onnxruntime.ORTModel`] کو ماڈل کے ساتھ فوری بنائیں اور `export=True` سیٹ کریں۔ ONNX ماڈل کو [`~optimum.onnxruntime.ORTModel.save_pretrained`] کے ساتھ محفوظ کریں۔

```py
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

ort_model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b", export=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")

ort_model.save_pretrained("onnx/")
tokenizer.save_pretrained("onnx/")
```
