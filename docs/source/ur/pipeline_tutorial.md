<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# پائپ لائن

[`Pipeline`] ایک سادہ لیکن طاقتور انفرنس API ہے جو Hugging Face [Hub](https://hf.co/models) سے کسی بھی ماڈل کے ساتھ مشین لرننگ کے مختلف کاموں کے لیے آسانی سے دستیاب ہے۔

[`Pipeline`] کو ٹاسک کے مخصوص پیرامیٹرز کے ساتھ اپنے کام کے مطابق بنائیں جیسے میٹنگ کے نوٹوں کو نقل کرنے کے لیے خودکار اسپیچ ریکگنیشن (ASR) پائپ لائن میں ٹائم اسٹیمپ شامل کرنا۔ [`Pipeline`] GPUs، Apple Silicon، اور Accelerate انفرنس میں آدھے درست وزن کو سپورٹ کرتا ہے اور میموری کو محفوظ کرتا ہے۔

<Youtube id=tiZFewofSLM/>

Transformers میں دو پائپ لائن کلاسز ہیں، ایک عام [`Pipeline`] اور بہت سی انفرادی ٹاسک مخصوص پائپ لائنیں جیسے [`TextGenerationPipeline`]۔ [`Pipeline`] میں `task` پیرامیٹر میں ٹاسک شناخت کنندہ کو ترتیب دے کر ان انفرادی پائپ لائنوں کو لوڈ کریں۔ آپ ان کے API دستاویزات میں ہر پائپ لائن کے لیے ٹاسک شناخت کنندہ تلاش کر سکتے ہیں۔

ہر کام کو ڈیفالٹ پہلے سے تربیت یافتہ ماڈل اور پری پروسیسر استعمال کرنے کے لیے ترتیب دیا گیا ہے، لیکن اگر آپ کوئی مختلف ماڈل استعمال کرنا چاہتے ہیں تو اسے `model` پیرامیٹر کے ساتھ اوور رائڈ کیا جا سکتا ہے۔

مثال کے طور پر، [`TextGenerationPipeline`] کو [Gemma 2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2) کے ساتھ استعمال کرنے کے لیے، `task="text-generation"` اور `model="google/gemma-2-2b"` سیٹ کریں۔

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

جب آپ کے پاس ایک سے زیادہ ان پٹ ہوں تو انہیں فہرست کے طور پر پاس کریں۔

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

یہ گائیڈ آپ کو [`Pipeline`] سے متعارف کرائے گا، اس کی خصوصیات کو ظاہر کرے گا، اور اس کے مختلف پیرامیٹرز کو ترتیب دینے کا طریقہ دکھائے گا۔

## کام

[`Pipeline`] مختلف طریقوں میں مشین لرننگ کے بہت سے کاموں کے ساتھ مطابقت رکھتا ہے۔ پائپ لائن میں ایک مناسب ان پٹ پاس کریں اور یہ باقی کو سنبھال لے گی۔

مختلف کاموں اور طریقوں کے لیے [`Pipeline`] استعمال کرنے کے طریقے کی کچھ مثالیں یہ ہیں۔

<hfoptions id="tasks">
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## پیرامیٹرز

کم از کم، [`Pipeline`] کو صرف ایک ٹاسک شناخت کنندہ، ماڈل، اور مناسب ان پٹ کی ضرورت ہوتی ہے۔ لیکن پائپ لائن کو ترتیب دینے کے لیے بہت سے پیرامیٹرز دستیاب ہیں، کام کے لیے مخصوص پیرامیٹرز سے لے کر کارکردگی کو بہتر بنانے تک۔

یہ سیکشن آپ کو کچھ زیادہ اہم پیرامیٹرز سے متعارف کراتا ہے۔

### ڈیوائس

[`Pipeline`] ہارڈ ویئر کی بہت سی اقسام کے ساتھ مطابقت رکھتا ہے، بشمول GPUs، CPUs، Apple Silicon، اور مزید۔ ہارڈ ویئر کی قسم کو `device` پیرامیٹر کے ساتھ کنفیگر کریں۔ پہلے سے طے شدہ طور پر، جب `device` کو سیٹ کیے بغیر چھوڑ دیا جاتا ہے، [`Pipeline`] خود بخود ماڈل کو پہلے دستیاب ایکسلریٹر (CUDA GPU, Apple Silicon MPS, XPU, ...) پر رکھتا ہے اور صرف CPU پر واپس آتا ہے جب کوئی دستیاب نہ ہو۔ اسے CPU پر چلانے کے لیے مجبور کرنے کے لیے `device="cpu"` پاس کریں۔

<hfoptions id="device">
<hfoption id="GPU">

GPU پر [`Pipeline`] چلانے کے لیے، `device` کو متعلقہ CUDA ڈیوائس آئی ڈی پر سیٹ کریں۔ مثال کے طور پر، `device=0` پہلے GPU پر چلتا ہے۔

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

آپ تقسیم شدہ تربیت کے لیے ایک لائبریری [Accelerate](https://hf.co/docs/accelerate/index) کو بھی خود بخود منتخب کرنے کی اجازت دے سکتے ہیں کہ مناسب ڈیوائس پر ماڈل وزن کو کیسے لوڈ اور اسٹور کیا جائے۔ یہ خاص طور پر مفید ہے اگر آپ کے پاس متعدد آلات ہیں۔ Accelerate پہلے تیز ترین ڈیوائس پر ماڈل وزن کو لوڈ اور اسٹور کرتا ہے، اور پھر ضرورت کے مطابق وزن کو دوسرے آلات (CPU، ہارڈ ڈرائیو) میں منتقل کرتا ہے۔ `device_map="auto"` سیٹ کریں تاکہ Accelerate ڈیوائس کا انتخاب کریں۔

> [!TIP] > یقینی بنائیں کہ [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) انسٹال ہے۔ >
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

Apple سلکان پر [`Pipeline`] چلانے کے لیے، `device="mps"` سیٹ کریں۔

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

<a id='batch-inference'></a>
### بیچ انفرنس

[`Pipeline`] `batch_size` پیرامیٹر کے ساتھ ان پٹ کے بیچوں پر بھی کارروائی کر سکتا ہے۔ بیچ انفرنس رفتار کو بہتر بنا سکتا ہے، خاص طور پر GPU پر، لیکن اس کی ضمانت نہیں ہے۔ دیگر متغیرات جیسے ہارڈ ویئر، ڈیٹا، اور ماڈل خود اس بات پر اثر انداز ہو سکتے ہیں کہ آیا بیچ انفرنس رفتار کو بہتر بناتا ہے۔ اس وجہ سے، بیچ انفرنس بطور ڈیفالٹ غیر فعال ہے۔

نیچے دی گئی مثال میں، جب 4 ان پٹ ہوتے ہیں اور `batch_size` کو 2 پر سیٹ کیا جاتا ہے، [`Pipeline`] ایک وقت میں ماڈل کو 2 ان پٹ کا ایک بیچ دیتا ہے۔

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

بیچ انفرنس کے لیے ایک اور اچھا استعمال کیس [`Pipeline`] میں ڈیٹا کو سٹریم کرنے کے لیے ہے۔

```py
from transformers import pipeline
from accelerate import Accelerator
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = Accelerator().device

# KeyDataset is a utility that returns the item in the dict returned by the dataset
dataset = load_dataset("stanfordnlp/imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

اس بات کا تعین کرنے کے لیے کہ آیا بیچ انفرنس کارکردگی کو بہتر بنانے میں مدد کر سکتا ہے، درج ذیل عام اصولوں کو ذہن میں رکھیں۔

1. یقینی طور پر جاننے کا واحد طریقہ یہ ہے کہ آپ اپنے ماڈل، ڈیٹا اور ہارڈ ویئر کی کارکردگی کی پیمائش کریں۔
2. اگر آپ تاخیر سے مجبور ہیں تو انفرنس بیچ نہ کریں (مثال کے طور پر ایک لائیو انفرنس پروڈکٹ)۔
3. اگر آپ CPU استعمال کر رہے ہیں تو انفرنس بیچ نہ کریں۔
4. اگر آپ اپنے ڈیٹا کا `sequence_length` نہیں جانتے ہیں تو انفرنس بیچ نہ کریں۔ کارکردگی کی پیمائش کریں، بار بار `sequence_length` میں شامل کریں، اور ناکامیوں سے بازیافت کرنے کے لیے آؤٹ آف میموری (OOM) چیک شامل کریں۔
5. اگر آپ کا `sequence_length` ریگولر ہے تو بیچ انفرنس کریں، اور اسے اس وقت تک آگے بڑھاتے رہیں جب تک کہ آپ OOM کی خرابی تک نہ پہنچ جائیں۔ GPU جتنا بڑا ہوگا، اتنا ہی مددگار بیچ انفرنس ہوگا۔
6. اس بات کو یقینی بنائیں کہ اگر آپ بیچ انفرنس کرنے کا فیصلہ کرتے ہیں تو آپ OOM کی غلطیوں کو سنبھال سکتے ہیں۔

### ٹاسک کے لیے مخصوص پیرامیٹرز

[`Pipeline`] کسی بھی پیرامیٹرز کو قبول کرتا ہے جو ہر انفرادی ٹاسک پائپ لائن کے ذریعہ تعاون یافتہ ہیں۔ یہ دیکھنے کے لیے کہ کس قسم کے پیرامیٹرز دستیاب ہیں ہر انفرادی ٹاسک پائپ لائن کو چیک کرنا یقینی بنائیں۔ اگر آپ کو کوئی ایسا پیرامیٹر نہیں ملتا جو آپ کے استعمال کے معاملے کے لیے کارآمد ہو، تو براہ کرم اس کی درخواست کرنے کے لیے GitHub [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) کھولیں۔

ذیل کی مثالیں کام کے لیے دستیاب کچھ پیرامیٹرز کو ظاہر کرتی ہیں۔

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

جب ہر لفظ بولا گیا تو واپس کرنے کے لیے `return_timestamps="word"` پیرامیٹر کو [`Pipeline`] پر منتقل کریں۔

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamps="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

</hfoption>
<hfoption id="text generation">

مکمل متن (پرامپٹ اور جنریٹڈ ٹیکسٹ) کی بجائے صرف تیار کردہ متن کو واپس کرنے کے لیے `return_full_text=False` سے [`Pipeline`] تک جائیں۔

[`~TextGenerationPipeline.__call__`] [`~GenerationMixin.generate`] طریقہ سے اضافی مطلوبہ الفاظ کے دلائل کی بھی حمایت کرتا ہے۔ ایک سے زیادہ تیار کردہ ترتیب واپس کرنے کے لیے، `num_return_sequences` کو 1 سے بڑی قدر پر سیٹ کریں۔

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## چنک بیچنگ

کچھ ایسی مثالیں ہیں جہاں آپ کو ٹکڑوں میں ڈیٹا پر کارروائی کرنے کی ضرورت ہے۔

- کچھ ڈیٹا کی اقسام کے لیے، ایک ہی ان پٹ (مثال کے طور پر، واقعی ایک لمبی آڈیو فائل) پر کارروائی کرنے سے پہلے اسے متعدد حصوں میں تقسیم کرنے کی ضرورت پڑ سکتی ہے۔
- کچھ کاموں کے لیے، جیسے زیرو شاٹ کی درجہ بندی یا سوال کا جواب دینا، ایک ان پٹ کو متعدد فارورڈ پاسز کی ضرورت ہو سکتی ہے جو `batch_size` پیرامیٹر کے ساتھ مسائل پیدا کر سکتے ہیں۔

[ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) کلاس کو استعمال کے ان معاملات کو سنبھالنے کے لیے ڈیزائن کیا گیا ہے۔ پائپ لائن کی دونوں کلاسیں ایک ہی طرح سے استعمال ہوتی ہیں، لیکن چونکہ [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) بیچنگ کو خود بخود ہینڈل کر سکتی ہے، اس لیے آپ کو اپنے ان پٹ ٹرگر کو فارورڈ پاس کرنے کی تعداد کے بارے میں فکر کرنے کی ضرورت نہیں ہے۔ اس کے بجائے، آپ ان پٹ سے آزادانہ طور پر `batch_size` کو بہتر بنا سکتے ہیں۔

ذیل کی مثال دکھاتی ہے کہ یہ [`Pipeline`] سے کیسے مختلف ہے۔

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## بڑے ڈیٹا سیٹس

بڑے ڈیٹا سیٹس کے ساتھ انفرنس کے لیے، آپ خود ڈیٹا سیٹ پر براہ راست اعادہ کر سکتے ہیں۔ یہ پورے ڈیٹاسیٹ کے لیے فوری طور پر میموری کو مختص کرنے سے گریز کرتا ہے، اور آپ کو خود بیچز بنانے کی فکر کرنے کی ضرورت نہیں ہے۔ `batch_size` پیرامیٹر کے ساتھ [بیچ انفرنس](#batch-inference) کو آزمائیں کہ آیا یہ کارکردگی کو بہتر بناتا ہے۔

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from accelerate import Accelerator
from datasets import load_dataset

device = Accelerator().device

dataset = load_dataset("stanfordnlp/imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

[`Pipeline`] والے بڑے ڈیٹا سیٹس پر انفرنس چلانے کے دیگر طریقوں میں ایک ایٹریٹر یا جنریٹر کا استعمال شامل ہے۔

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## بڑا ماڈلز

[Accelerate](https://hf.co/docs/accelerate/index) بڑے ماڈلز کو [`Pipeline`] کے ساتھ چلانے کے لیے کچھ اصلاح کو قابل بناتا ہے۔ یقینی بنائیں کہ Accelerate پہلے انسٹال ہے۔

```py
!pip install -U accelerate
```

`device_map="auto"` ترتیب ماڈل کو تیز ترین ڈیوائسز (GPUs) میں خود بخود تقسیم کرنے کے لیے مفید ہے اگر دستیاب ہو تو دوسرے سست آلات (CPU، ہارڈ ڈرائیو) پر بھیجنے سے پہلے۔

[`Pipeline`] آدھے درست وزن (torch.float16) کو سپورٹ کرتا ہے، جو نمایاں طور پر تیز ہو سکتا ہے اور میموری کو بچا سکتا ہے۔ کارکردگی کا نقصان زیادہ تر ماڈلز کے لیے نہ ہونے کے برابر ہے، خاص طور پر بڑے کے لیے۔ اگر آپ کا ہارڈویئر اس کی حمایت کرتا ہے، تو آپ مزید رینج کے لیے torch.bfloat16 کو فعال کر سکتے ہیں۔

> [!TIP] > ان پٹ کو اندرونی طور پر torch.float16 میں تبدیل کیا جاتا ہے اور یہ PyTorch بیک اینڈ کے ساتھ صرف ماڈلز کے لیے کام کرتا ہے۔

آخر میں، میموری کے استعمال کو مزید کم کرنے کے لیے [`Pipeline`] کوانٹائزڈ ماڈلز بھی قبول کرتا ہے۔ یقینی بنائیں کہ آپ کے پاس پہلے [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) لائبریری انسٹال ہے، اور پھر پائپ لائن میں `quantization_config` کو `model_kwargs` شامل کریں۔

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
