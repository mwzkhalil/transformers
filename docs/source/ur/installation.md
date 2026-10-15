<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# تنصیب

Transformers، [PyTorch](https://pytorch.org/get-started/locally/) کے ساتھ کام کرتا ہے۔ اسے Python 3.10+ اور PyTorch 2.6+ پر آزمایا گیا ہے۔

## ورچوئل ماحول

[uv](https://docs.astral.sh/uv/) ایک نہایت تیز، Rust پر مبنی Python پیکیج اور پراجیکٹ مینیجر ہے۔ مختلف پراجیکٹس کی dependencies سنبھالنے اور ان کے درمیان مطابقت کے مسائل سے بچنے کے لیے اسے بطورِ طے شدہ [ورچوئل ماحول](https://docs.astral.sh/uv/pip/environments/) درکار ہوتا ہے۔

اسے [pip](https://pip.pypa.io/en/stable/) کی جگہ براہِ راست استعمال کیا جا سکتا ہے۔ اگر آپ pip استعمال کرنا چاہتے ہیں تو نیچے دی گئی کمانڈز سے `uv` ہٹا دیں۔

> [!TIP]
> uv کی تنصیب کے لیے اس کی [رہنمائی](https://docs.astral.sh/uv/guides/install-python/) دیکھیں۔

Transformers نصب کرنے کے لیے ورچوئل ماحول بنائیں۔

```bash
uv venv .env
source .env/bin/activate
```

## Python

درج ذیل کمانڈ سے Transformers نصب کریں۔

[uv](https://docs.astral.sh/uv/) ایک تیز رفتار، Rust پر مبنی Python پیکیج اور پراجیکٹ مینیجر ہے۔


<hfoptions id="installation">
<hfoption id="CUDA">

NVIDIA GPU ‏(CUDA) کے ساتھ PyTorch سمیت Transformers نصب کرنے کے لیے [PyTorch](https://pytorch.org/get-started/locally) کی ہدایات کے مطابق مناسب CUDA ڈرائیورز نصب کریں۔

یہ جانچنے کے لیے کہ آپ کا سسٹم NVIDIA GPU کو پہچانتا ہے یا نہیں، درج ذیل کمانڈ چلائیں۔

```bash
nvidia-smi
```

```bash
uv pip install "transformers[torch]"
```

</hfoption>
<hfoption id="NVIDIA Spark (ARM64)">

NVIDIA Spark جیسے ARM64 ڈیوائسز پر PyTorch کے ساتھ Transformers نصب کرنے کے لیے NVIDIA PyPI انڈیکس سے PyTorch نصب کریں۔ ان ڈیوائسز کو PyTorch کی NVIDIA ARM64 builds درکار ہیں، جو عام PyPI انڈیکس یا PyTorch کے معیاری wheel انڈیکس پر دستیاب نہیں۔

یہ جانچنے کے لیے کہ آپ کا سسٹم NVIDIA GPU کو پہچانتا ہے یا نہیں، درج ذیل کمانڈ چلائیں۔

```bash
nvidia-smi
```

NVIDIA PyPI انڈیکس سے PyTorch نصب کریں، پھر Transformers نصب کریں۔

```bash
uv pip install torch --index-url https://pypi.nvidia.com
uv pip install transformers
```

</hfoption>
<hfoption id="CPU">

صرف CPU کے لیے Transformers نصب کرنے کا طریقہ یہ ہے۔

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

</hfoption>
<hfoption id="Intel GPU (XPU)">

Intel GPU ‏(XPU) کے لیے PyTorch سمیت Transformers نصب کرنے سے پہلے [Intel GPU ‏(XPU) کے PyTorch ڈرائیورز](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html) نصب کریں اور PyTorch کا XPU انڈیکس شامل کریں۔

یہ جانچنے کے لیے کہ سسٹم Intel GPU کو پہچانتا ہے یا نہیں، [تصدیق کی ہدایات](https://dgpu-docs.intel.com/driver/verification.html) دیکھیں اور درج ذیل کمانڈ چلائیں۔

```bash
xpu-smi
```

```bash
uv pip install "transformers[torch]" --extra-index-url https://download.pytorch.org/whl/xpu
```

</hfoption>
</hfoptions>

یہ جانچنے کے لیے کہ تنصیب کامیاب ہوئی ہے، درج ذیل کمانڈ چلائیں۔ دیے گئے متن کے لیے اسے ایک label اور score واپس کرنا چاہیے۔

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### سورس سے تنصیب

سورس سے تنصیب لائبریری کے *stable* ورژن کے بجائے *latest* ورژن کو نصب کرتی ہے۔ اس طرح Transformers کی تازہ ترین تبدیلیاں ملتی ہیں، جو نئی خصوصیات آزمانے یا stable ورژن میں ابھی تک درست نہ ہونے والے کسی bug کو ٹھیک کرنے کے لیے مفید ہیں۔

اس کا نقصان یہ ہے کہ تازہ ترین ورژن ہمیشہ مستحکم نہیں ہوتا۔ اگر کوئی مسئلہ درپیش ہو تو براہِ کرم [GitHub Issue](https://github.com/huggingface/transformers/issues) کھولیں تاکہ ہم اسے جلد درست کر سکیں۔

درج ذیل کمانڈ سے سورس سے تنصیب کریں۔

```bash
uv pip install git+https://github.com/huggingface/transformers
```

درج ذیل کمانڈ سے تنصیب کی کامیابی جانچیں۔ دیے گئے متن کے لیے اسے ایک label اور score واپس کرنا چاہیے۔

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### قابلِ تدوین تنصیب

اگر آپ مقامی طور پر Transformers پر کام کر رہے ہیں تو [قابلِ تدوین تنصیب](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) مفید ہے۔ اس میں Transformers کی فائلیں نقل کرنے کے بجائے آپ کی مقامی کاپی کو Transformers کی [ریپوزٹری](https://github.com/huggingface/transformers) سے جوڑا جاتا ہے۔ فائلیں Python کے import path میں شامل ہو جاتی ہیں۔

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> آپ کو Transformers کا مقامی فولڈر برقرار رکھنا ہوگا۔

Transformers کی مقامی کاپی کو مرکزی ریپوزٹری کی تازہ ترین تبدیلیوں سے اپ ڈیٹ کرنے کے لیے یہ کمانڈ چلائیں۔

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#) مختلف زبانوں کے پیکیجز کا مینیجر ہے۔ اپنے نئے ورچوئل ماحول میں [conda-forge](https://anaconda.org/conda-forge/transformers) چینل سے Transformers نصب کریں۔

```bash
conda install conda-forge::transformers
```

## ترتیب دیں

تنصیب کے بعد Transformers کی cache کا مقام مقرر کیا جا سکتا ہے یا لائبریری کو آف لائن استعمال کے لیے ترتیب دیا جا سکتا ہے۔

<a id='cache-directory'></a>
### cache ڈائریکٹری

جب آپ [`~PreTrainedModel.from_pretrained`] سے پہلے سے تربیت یافتہ ماڈل لوڈ کرتے ہیں تو ماڈل Hub سے ڈاؤن لوڈ ہوتا ہے اور مقامی cache میں محفوظ ہو جاتا ہے۔

اگر آپ commit hash دیں تو Transformers اس commit کی فائلوں کے لیے مقامی cache استعمال کرتا ہے اور ہر فائل کے لیے Hub کو دوبارہ نہیں جانچتا (ان فائلوں کے لیے بھی نہیں جن کے نہ ہونے کا پہلے سے علم ہو)۔ جب کوئی revision نہیں دیا جاتا تو یہی طے شدہ رویہ ہوتا ہے۔

اگر آپ branch یا tag دیں تو Transformers لوڈنگ کے آغاز پر commit معلوم کرنے کے لیے ایک بار Hub سے رابطہ کرتا ہے، پھر اسی طرح cache استعمال کرتا ہے۔

shell environment variable ‏`HF_HUB_CACHE` کی طے شدہ ڈائریکٹری `~/.cache/huggingface/hub` ہے۔ Windows پر طے شدہ ڈائریکٹری `C:\Users\username\.cache\huggingface\hub` ہے۔

ماڈل کو کسی دوسری ڈائریکٹری میں cache کرنے کے لیے درج ذیل shell environment variables میں path بدلیں (ترجیحی ترتیب کے مطابق)۔

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (طے شدہ)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (صرف اس صورت میں جب `HF_HOME` مقرر نہ ہو)

<a id='offline-mode'></a>
### آف لائن موڈ

Transformers کو آف لائن یا firewall والے ماحول میں استعمال کرنے کے لیے ضروری فائلیں پہلے ہی ڈاؤن لوڈ کرکے cache میں رکھنی ہوتی ہیں۔ [`~huggingface_hub.snapshot_download`] طریقے سے Hub سے ماڈل کی ریپوزٹری ڈاؤن لوڈ کریں۔

> [!TIP]
> فائلیں ڈاؤن لوڈ کرنے کے مزید طریقوں کے لیے Hub کی [Download files](https://hf.co/docs/huggingface_hub/guides/download) رہنمائی دیکھیں۔ آپ مخصوص revisions سے فائلیں ڈاؤن لوڈ کر سکتے ہیں، CLI استعمال کر سکتے ہیں اور ڈاؤن لوڈ کے لیے فائلیں منتخب بھی کر سکتے ہیں۔

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

ماڈل لوڈ کرتے وقت Hub کو HTTP درخواستیں بھیجنے سے روکنے کے لیے environment variable ‏`HF_HUB_OFFLINE=1` مقرر کریں۔

صرف cache میں موجود فائلیں لوڈ کرنے کا ایک اور طریقہ [`~PreTrainedModel.from_pretrained`] میں `local_files_only=True` دینا ہے۔

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```

آف لائن موڈ (یا `local_files_only=True`) میں بھی branch یا tag کو commit میں بدلا جا سکتا ہے، اگر پہلے کسی آن لائن لوڈ کے دوران اس کی mapping cache میں محفوظ ہو چکی ہو۔ اگر mapping کبھی محفوظ نہ ہوئی ہو تو Transformers آپ کی دی ہوئی branch یا tag برقرار رکھتا ہے اور معمول کے آف لائن لوڈ کے مطابق آگے بڑھتا ہے۔ نتیجے میں cache hit یا گم شدہ فائل کی وہی errors ملتی ہیں جو عام آف لائن لوڈ میں ملتیں۔

اگلے مرحلے کے لیے [فوری آغاز](./quicktour) دیکھیں، یا [Transformers کے تعارف](./index) پر واپس جائیں۔
