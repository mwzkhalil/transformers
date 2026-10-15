<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# تصویر کی کیپشننگ

[[open-in-colab]]

امیج کیپشننگ ایک دی گئی تصویر کے لیے کیپشن کی پیشن گوئی کرنے کا کام ہے۔ اس کی عام حقیقی دنیا کی ایپلی کیشنز میں بصارت سے محروم لوگوں کی مدد کرنا شامل ہے جو انہیں مختلف حالات میں نیویگیٹ کرنے میں مدد دے سکتی ہے۔ لہذا، تصویر کیپشننگ لوگوں کے لیے تصاویر کو بیان کرکے مواد کی رسائی کو بہتر بنانے میں مدد کرتی ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

* فائن ٹیون ایک تصویر کیپشن ماڈل۔
* انفرنس کے لیے فائن ٹیونڈ ماڈل استعمال کریں۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate -q
pip install jiwer -q
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Pokémon BLIP کیپشن ڈیٹاسیٹ لوڈ کریں۔

ڈیٹا سیٹ لوڈ کرنے کے لیے 🤗 ڈیٹا سیٹ لائبریری کا استعمال کریں جو کہ {image-caption} جوڑوں پر مشتمل ہو۔ PyTorch میں اپنا تصویری کیپشننگ ڈیٹاسیٹ بنانے کے لیے، آپ [اس نوٹ بک](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb) کی پیروی کر سکتے ہیں۔

```python
from datasets import load_dataset

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds
```

```bash
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 833
    })
})
```

ڈیٹاسیٹ میں دو خصوصیات ہیں، `image` اور `text`۔

<Tip>

بہت سے امیج کیپشننگ ڈیٹاسیٹس میں فی تصویر ایک سے زیادہ کیپشن ہوتے ہیں۔ ان صورتوں میں، ایک مشترکہ حکمت عملی یہ ہے کہ تربیت کے دوران دستیاب لوگوں کے درمیان تصادفی طور پر ایک کیپشن کا نمونہ لیا جائے۔

</Tip>

ڈیٹا سیٹ کے ٹرین اسپلٹ کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```python
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```

آئیے تربیتی سیٹ سے کچھ نمونوں کا تصور کرتے ہیں۔

```python
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")


sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_training_images_image_cap.png" alt="Sample training images"/>
</div>

## ڈیٹا سیٹ کو پہلے سے پروسیس کریں۔

چونکہ ڈیٹاسیٹ میں دو طریقوں (تصویر اور متن) ہیں، اس لیے پری پروسیسنگ پائپ لائن امیجز اور کیپشنز کو پہلے سے پروسیس کرے گی۔

ایسا کرنے کے لیے، ماڈل سے منسلک پروسیسر کی کلاس لوڈ کریں جس کے بارے میں آپ فائن ٹیون ہونے والے ہیں۔

```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```

پروسیسر تصویر کو اندرونی طور پر پہلے سے پروسیس کرے گا (جس میں سائز تبدیل کرنا، اور پکسل اسکیلنگ شامل ہے) اور کیپشن کو ٹوکنائز کرے گا۔

```python
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)
```

ڈیٹا سیٹ کے تیار ہونے کے ساتھ، آپ اب ماڈل کو فائن ٹیوننگ کے لیے سیٹ کر سکتے ہیں۔

## بیس ماڈل لوڈ کریں

["microsoft/git-base"](https://huggingface.co/microsoft/git-base) کو [`AutoModelForCausalLM`] آبجیکٹ میں لوڈ کریں۔

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

## اندازہ لگانا

تصویری کیپشن ماڈلز کا اندازہ عام طور پر [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) یا [Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer) سے کیا جاتا ہے۔ اس گائیڈ کے لیے، آپ ورڈ ایرر ریٹ (WER) استعمال کریں گے۔

ہم ایسا کرنے کے لیے 🤗 Evaluate لائبریری کا استعمال کرتے ہیں۔ WER کی ممکنہ حدود اور دیگر گٹچز کے لیے، [یہ گائیڈ](https://huggingface.co/spaces/evaluate-metric/wer) دیکھیں۔

```python
from evaluate import load
import torch

wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
```

## ٹرین

اب، آپ فائن ٹیوننگ ماڈل شروع کرنے کے لیے تیار ہیں۔ آپ اس کے لیے 🤗 [`Trainer`] استعمال کریں گے۔

سب سے پہلے، [`TrainingArguments`] کا استعمال کرتے ہوئے تربیتی دلائل کی وضاحت کریں۔

```python
from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)
```

پھر انہیں ڈیٹا سیٹس اور ماڈل کے ساتھ 🤗 ٹرینر کے پاس بھیجیں۔

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

ٹریننگ شروع کرنے کے لیے، بس [`~Trainer.train`] کو [`Trainer`] آبجیکٹ پر کال کریں۔

```python
trainer.train()
```

آپ کو تربیت کے نقصان کو آسانی سے گرتا ہوا دیکھنا چاہئے جیسے جیسے تربیت کی ترقی ہوتی ہے۔

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```python
trainer.push_to_hub()
```

## انفرنس

ماڈل کو جانچنے کے لیے `test_ds` سے نمونہ کی تصویر لیں۔

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/test_image_image_cap.png" alt="Test image"/>
</div>

ماڈل کے لیے تصویر تیار کریں۔

```python
from accelerate import Accelerator

device = Accelerator().device
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```

[`generate`] کال کریں اور پیشین گوئیوں کو ڈی کوڈ کریں۔

```python
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```

```bash
a drawing of a pink and blue pokemon
```

ایسا لگتا ہے کہ ٹھیک ٹیونڈ ماڈل نے ایک بہت اچھا کیپشن تیار کیا ہے!
