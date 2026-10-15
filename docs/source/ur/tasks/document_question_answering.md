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

# دستاویزی سوال جواب

[[open-in-colab]]

دستاویزی سوال کا جواب دینا، جسے دستاویزی بصری سوال کا جواب بھی کہا جاتا ہے، ایک ایسا کام ہے جس میں دستاویز کی تصاویر کے بارے میں پوچھے گئے سوالات کے جوابات فراہم کرنا شامل ہے۔ ماڈلز اس کام کو سپورٹ کرنے کا ان پٹ عام طور پر ایک تصویر اور سوال کا مجموعہ ہوتا ہے، اور آؤٹ پٹ ایک جواب ہوتا ہے جس کا اظہار قدرتی زبان میں ہوتا ہے۔ یہ ماڈلز متعدد طریقوں کا استعمال کرتے ہیں، بشمول متن، الفاظ کی پوزیشنیں (باؤنڈنگ بکس) اور خود تصویر۔

یہ گائیڈ وضاحت کرتا ہے کہ کیسے:

- فائن ٹیون [DocVQA ڈیٹا سیٹ](https://huggingface.co/docs/transformers/main/en/model_doc/layoutlmv2) پر لے آؤٹLMv2](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut)۔
- انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/image-to-text) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

LayoutLMv2 جواب کے آغاز اور اختتامی ٹوکن کی پوزیشنوں کی پیشین گوئی کرنے کے لیے، ٹوکنز کی آخری پوشیدہ حالتوں کے اوپر ایک سوال کا جواب دینے والا سر شامل کرکے دستاویز کے سوال جواب دینے کے کام کو حل کرتا ہے۔ دوسرے لفظوں میں، مسئلہ کو استخراجی سوال کے جواب کے طور پر سمجھا جاتا ہے: سیاق و سباق کو دیکھتے ہوئے، معلومات کا کون سا حصہ سوال کا جواب دیتا ہے۔ سیاق و سباق OCR انجن کے آؤٹ پٹ سے آتا ہے، یہاں یہ گوگل کا ٹیسریکٹ ہے۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں۔ لے آؤٹLMv2 کا انحصار detectron2، torchvision اور tesseract پر ہے۔

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

ایک بار جب آپ تمام انحصار انسٹال کر لیں تو اپنا رن ٹائم دوبارہ شروع کریں۔

ہم آپ کی حوصلہ افزائی کرتے ہیں کہ اپنا ماڈل کمیونٹی کے ساتھ شیئر کریں۔ اسے 🤗 حب پر اپ لوڈ کرنے کے لیے اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

آئیے کچھ عالمی متغیرات کی وضاحت کرتے ہیں۔

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## ڈیٹا لوڈ کریں۔

اس گائیڈ میں ہم پہلے سے تیار شدہ DocVQA کا ایک چھوٹا نمونہ استعمال کرتے ہیں جو آپ کو 🤗 Hub پر مل سکتا ہے۔ اگر آپ مکمل DocVQA ڈیٹاسیٹ استعمال کرنا چاہتے ہیں، تو آپ اسے [DocVQA ہوم پیج](https://rrc.cvc.uab.es/?ch=17) پر رجسٹر اور ڈاؤن لوڈ کر سکتے ہیں۔ اگر آپ ایسا کرتے ہیں تو، اس گائیڈ کے ساتھ آگے بڑھنے کے لیے [ 🤗 ڈیٹا سیٹ](https://huggingface.co/docs/datasets/loading#local-and-remote-files) میں فائلوں کو کیسے لوڈ کیا جائے کو چیک کریں۔

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 200
    })
})
```

جیسا کہ آپ دیکھ سکتے ہیں، ڈیٹاسیٹ پہلے ہی ٹرین اور ٹیسٹ سیٹ میں تقسیم ہے۔ خصوصیات سے اپنے آپ کو واقف کرنے کے لیے ایک بے ترتیب مثال پر ایک نظر ڈالیں۔

```py
>>> dataset["train"].features
```

یہاں وہ ہے جو انفرادی فیلڈز کی نمائندگی کرتے ہیں:

* `id`: مثال کی شناخت
* `image`: دستاویز کی تصویر پر مشتمل ایک PIL.Image.Image آبجیکٹ
* `query`: سوال کی تار - کئی زبانوں میں قدرتی زبان سے پوچھا گیا سوال
* `answers`: انسانی تشریح کاروں کے ذریعہ فراہم کردہ درست جوابات کی فہرست
* `words` اور `bounding_boxes`: OCR کے نتائج، جنہیں ہم یہاں استعمال نہیں کریں گے۔
* `answer`: ایک مختلف ماڈل سے مماثل جواب جسے ہم یہاں استعمال نہیں کریں گے۔

آئیے صرف انگریزی کے سوالات چھوڑتے ہیں، اور `answer` خصوصیت کو چھوڑ دیتے ہیں جو بظاہر کسی اور ماڈل کی پیشین گوئیوں پر مشتمل ہوتا ہے۔ ہم تشریح کاروں کے فراہم کردہ سیٹ سے پہلے جوابات بھی لیں گے۔ متبادل طور پر، آپ تصادفی طور پر اس کا نمونہ لے سکتے ہیں۔

```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

نوٹ کریں کہ LayoutLMv2چیک پوائنٹجو ہم اس گائیڈ میں استعمال کرتے ہیں اس کے ساتھ تربیت دی گئی ہے۔`max_position_embeddings = 512`(آپ یہ معلومات [چیک پوائنٹکی`config.json`فائل](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18))۔ ہم مثالوں کو تراش سکتے ہیں لیکن اس صورت حال سے بچنے کے لیے جہاں جواب کسی بڑی دستاویز کے آخر میں ہو اور ختم ہو جائے، یہاں ہم ان چند مثالوں کو ہٹا دیں گے جہاں ایمبیڈنگ 512 سے زیادہ طویل ہونے کا امکان ہے۔ اگر آپ کے ڈیٹا سیٹ میں زیادہ تر دستاویزات طویل ہیں، تو آپ سلائیڈنگ ونڈو کی حکمت عملی کو لاگو کر سکتے ہیں - یہ نوٹ بک چیک کریں](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb)تفصیلات کے لیے

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

اس وقت آئیے اس ڈیٹاسیٹ سے OCR خصوصیات کو بھی ہٹا دیں۔ یہ فائن ٹیوننگ ایک مختلف ماڈل کے لیے OCR کا نتیجہ ہیں۔ اگر ہم انہیں استعمال کرنا چاہتے ہیں تو انہیں ابھی بھی کچھ پروسیسنگ کی ضرورت ہوگی، کیونکہ وہ ماڈل کی ان پٹ ضروریات سے میل نہیں کھاتے ہیں جو ہم اس گائیڈ میں استعمال کرتے ہیں۔ اس کے بجائے، ہم OCR اور ٹوکنائزیشن دونوں کے لیے اصل ڈیٹا پر [`LayoutLMv2Processor`] استعمال کر سکتے ہیں۔ اس طرح ہمیں ان پٹ ملیں گے جو ماڈل کے متوقع ان پٹ سے ملتے ہیں۔ اگر آپ تصویروں کو دستی طور پر پروسیس کرنا چاہتے ہیں، تو [`LayoutLMv2` ماڈل دستاویزات](https://huggingface.co/docs/transformers/main/en/model_doc/layoutlmv2) یہ جاننے کے لیے دیکھیں کہ ماڈل کس ان پٹ فارمیٹ کی توقع کرتا ہے۔

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

آخر میں، ڈیٹا کی تلاش مکمل نہیں ہوگی اگر ہم تصویر کی مثال کو نہیں دیکھتے۔

```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="DocVQA Image Example"/>
 </div>

## ڈیٹا کو پہلے سے پروسیس کریں۔

دستاویز سوال کا جواب دینے کا کام ایک ملٹی موڈل ٹاسک ہے، اور آپ کو یہ یقینی بنانا ہوگا کہ ہر موڈیلٹی کے ان پٹس کو ماڈل کی توقعات کے مطابق پہلے سے پروسیس کیا گیا ہے۔ آئیے [`LayoutLMv2Processor`] لوڈ کر کے شروع کریں، جو اندرونی طور پر ایک تصویری پروسیسر کو جوڑتا ہے جو تصویری ڈیٹا کو سنبھال سکتا ہے اور ایک ٹوکنائزر جو ٹیکسٹ ڈیٹا کو انکوڈ کر سکتا ہے۔

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### دستاویز کی تصاویر سے پہلے پروسیسنگ

سب سے پہلے، آئیے پروسیسر سے `image_processor` کی مدد سے ماڈل کے لیے دستاویز کی تصاویر تیار کریں۔ پہلے سے طے شدہ طور پر، امیج پروسیسر تصاویر کا سائز 224x224 کر دیتا ہے، اس بات کو یقینی بناتا ہے کہ ان کے پاس رنگین چینلز کا صحیح ترتیب ہے، الفاظ اور نارملائز باؤنڈنگ بکس حاصل کرنے کے لیے ٹیسریکٹ کے ساتھ OCR کا اطلاق ہوتا ہے۔ اس ٹیوٹوریل میں، یہ تمام ڈیفالٹس بالکل وہی ہیں جن کی ہمیں ضرورت ہے۔ ایک فنکشن لکھیں جو ڈیفالٹ امیج پروسیسنگ کو امیجز کے بیچ پر لاگو کرتا ہے اور OCR کے نتائج واپس کرتا ہے۔

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

اس پری پروسیسنگ کو پورے ڈیٹاسیٹ پر تیزی سے لاگو کرنے کے لیے، [`~datasets.Dataset.map`] استعمال کریں۔

```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### ٹیکسٹ ڈیٹا کی پری پروسیسنگ

ایک بار جب ہم نے تصویروں پر OCR لاگو کر دیا، تو ہمیں ڈیٹاسیٹ کے ٹیکسٹ حصے کو ماڈل کے لیے تیار کرنے کے لیے انکوڈ کرنا ہوگا۔ اس میں ان الفاظ اور خانوں کو تبدیل کرنا شامل ہے جو ہمیں پچھلے مرحلے میں ٹوکن لیول `input_ids`، `attention_mask`، `token_type_ids` اور `bbox` میں ملے تھے۔ متن کی پری پروسیسنگ کے لیے، ہمیں پروسیسر سے `tokenizer` کی ضرورت ہوگی۔

```py
>>> tokenizer = processor.tokenizer
```

اوپر بیان کردہ پری پروسیسنگ کے اوپر، ہمیں ماڈل کے لیے لیبلز بھی شامل کرنے کی ضرورت ہے۔ `xxxForQuestionAnswering` ماڈلز in 🤗 Transformers کے لیے، لیبلز `start_positions` اور `end_positions` پر مشتمل ہوتے ہیں، یہ بتاتے ہیں کہ کون سا ٹوکن شروع میں ہے اور کون سا جواب آخر میں ہے۔

چلو اس کے ساتھ شروع کرتے ہیں. ایک مددگار فنکشن کی وضاحت کریں جو ایک بڑی فہرست (الفاظ کی فہرست) میں سب لسٹ (جواب الفاظ میں تقسیم) تلاش کر سکے۔

یہ فنکشن ان پٹ کے طور پر دو فہرستیں لے گا،`words_list`اور`answer_list`. اس کے بعد اس کے اوپر اعادہ ہوگا۔`words_list`اور چیک کریں کہ آیا موجودہ لفظ میں`words_list`(words_list[i]) answer_list (answer_list[0]) کے پہلے لفظ کے برابر ہے اور اگر ذیلی فہرست`words_list`موجودہ لفظ سے شروع ہو رہا ہے اور اسی لمبائی کا`answer_list`برابر ہے`to answer_list`. اگر یہ شرط درست ہے، تو اس کا مطلب ہے کہ ایک مماثلت مل گئی ہے، اور فنکشن میچ، اس کا ابتدائی اشاریہ (idx)، اور اس کے اختتامی اشاریہ (idx + len(answer_list) - 1) کو ریکارڈ کرے گا۔ اگر ایک سے زیادہ مماثلت پائی جاتی ہے، تو فنکشن صرف پہلا ہی لوٹائے گا۔ اگر کوئی مماثلت نہیں ملتی ہے، فنکشن واپس آتا ہے (`None`، 0، اور 0)۔

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

یہ واضح کرنے کے لیے کہ یہ فنکشن کس طرح جواب کی پوزیشن کو تلاش کرتا ہے، آئیے اسے ایک مثال پر استعمال کرتے ہیں:

```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  Who is in  cc in this letter?
Words: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorming', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast', 'buming', 'cigarette.', '«novel', 'cigarette', 'constructions', 'that', 'permit', 'a', 'significant', 'reduction', 'iretobacco', 'weight', 'while', 'maintaining', 'smoking', 'mechanics', 'and', 'visual', 'characteristics.', 'higher', 'basis', 'weight', 'paper:', 'potential', 'reduction', 'in', 'tobacco', 'weight.', '«more', 'rigid', 'tobacco', 'column;', 'stiffing', 'agent', 'for', 'tobacco;', 'e.g.', 'starch', '*colored', 'tow', 'and', 'cigarette', 'papers;', 'seasonal', 'promotions,', 'e.g.', 'pastel', 'colored', 'cigarettes', 'for', 'easter', 'or', 'in', 'an', 'ebony', 'and', 'ivory', 'brand', 'containing', 'a', 'mixture', 'of', 'all', 'black', '(black', 'paper', 'and', 'tow)', 'and', 'ail', 'white', 'cigarettes.', '499150498']
Answer:  T.F. Riehl
start_index 17
end_index 18
```

مثالوں کو انکوڈ کرنے کے بعد، تاہم، وہ اس طرح نظر آئیں گے:

```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
[CLS] who is in cc in this letter? [SEP] wie baw brown & williamson tobacco corporation research & development ...
```

ہمیں انکوڈ شدہ ان پٹ میں جواب کی پوزیشن تلاش کرنے کی ضرورت ہوگی۔

* `token_type_ids` ہمیں بتاتا ہے کہ کون سے ٹوکن سوال کا حصہ ہیں، اور کون سے دستاویز کے الفاظ کا حصہ ہیں۔
* `tokenizer.cls_token_id` ان پٹ کے شروع میں خصوصی ٹوکن تلاش کرنے میں مدد کرے گا۔
* `word_ids` اصل `words` میں ملے جواب کو مکمل انکوڈ شدہ ان پٹ میں اسی جواب سے ملانے اور تعین کرنے میں مدد کرے گا
انکوڈ شدہ ان پٹ میں جواب کی شروعات/اختتام کی پوزیشن۔

اس کو ذہن میں رکھتے ہوئے، آئیے ڈیٹاسیٹ میں مثالوں کے بیچ کو انکوڈ کرنے کے لیے ایک فنکشن بنائیں:

```py
>>> def encode_dataset(examples, max_length=512):
...     questions = examples["question"]
...     words = examples["words"]
...     boxes = examples["boxes"]
...     answers = examples["answer"]

...     # encode the batch of examples and initialize the start_positions and end_positions
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []

...     # loop through the examples in the batch
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)

...         # find the position of the answer in example's words
...         words_example = [word.lower() for word in words[i]]
...         answer = answers[i]
...         match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())

...         if match:
...             # if match is found, use `token_type_ids` to find where words start in the encoding
...             token_type_ids = encoding["token_type_ids"][i]
...             token_start_index = 0
...             while token_type_ids[token_start_index] != 1:
...                 token_start_index += 1

...             token_end_index = len(encoding["input_ids"][i]) - 1
...             while token_type_ids[token_end_index] != 1:
...                 token_end_index -= 1

...             word_ids = encoding.word_ids(i)[token_start_index : token_end_index + 1]
...             start_position = cls_index
...             end_position = cls_index

...             # loop over word_ids and increase `token_start_index` until it matches the answer position in words
...             # once it matches, save the `token_start_index` as the `start_position` of the answer in the encoding
...             for id in word_ids:
...                 if id == word_idx_start:
...                     start_position = token_start_index
...                 else:
...                     token_start_index += 1

...             # similarly loop over `word_ids` starting from the end to find the `end_position` of the answer
...             for id in word_ids[::-1]:
...                 if id == word_idx_end:
...                     end_position = token_end_index
...                 else:
...                     token_end_index -= 1

...             start_positions.append(start_position)
...             end_positions.append(end_position)

...         else:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)

...     encoding["image"] = examples["image"]
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions

...     return encoding
```

اب جب کہ ہمارے پاس یہ پری پروسیسنگ فنکشن ہے، ہم پورے ڈیٹاسیٹ کو انکوڈ کر سکتے ہیں:

```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```

آئیے چیک کریں کہ انکوڈ شدہ ڈیٹاسیٹ کی خصوصیات کیسی نظر آتی ہیں:

```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'start_positions': Value(dtype='int64', id=None),
 'end_positions': Value(dtype='int64', id=None)}
```

## تشخیص

دستاویز کے سوال کے جوابات کی تشخیص کے لیے کافی مقدار میں پوسٹ پروسیسنگ کی ضرورت ہوتی ہے۔ اپنا زیادہ وقت ضائع کرنے سے بچنے کے لیے، یہ گائیڈ تشخیصی مرحلے کو چھوڑ دیتا ہے۔ [`Trainer`] اب بھی تربیت کے دوران تشخیصی نقصان کا حساب لگاتا ہے تاکہ آپ اپنی ماڈل کی کارکردگی کے بارے میں مکمل طور پر اندھیرے میں نہ ہوں۔ استخراجی سوالوں کے جوابات کا عام طور پر F1/ عین مطابق میچ کا استعمال کرتے ہوئے جائزہ لیا جاتا ہے۔ اگر آپ اسے خود نافذ کرنا چاہتے ہیں، تو الہام کے لیے Hugging Face کورس کا [سوال جواب دینے والا باب](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) دیکھیں۔

## ٹرین

مبارک ہو! آپ نے اس گائیڈ کے مشکل ترین حصے کو کامیابی کے ساتھ نیویگیٹ کر لیا ہے اور اب آپ اپنی ماڈل کو تربیت دینے کے لیے تیار ہیں۔ تربیت میں درج ذیل مراحل شامل ہیں:

* ماڈل کو [`AutoModelForDocumentQuestionAnswering`] کے ساتھ اسی چیک پوائنٹ کا استعمال کرتے ہوئے لوڈ کریں جیسا کہ پری پروسیسنگ میں ہے۔
* [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔
* مثالوں کو اکٹھا کرنے کے لیے ایک فنکشن کی وضاحت کریں، یہاں [`DefaultDataCollator`] ٹھیک کام کرے گا
* ماڈل، ڈیٹاسیٹ، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
* اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

[`TrainingArguments`] میں یہ بتانے کے لیے `output_dir` استعمال کریں کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے، اور جیسا کہ آپ مناسب دیکھیں ہائپر پیرامیٹر کو ترتیب دیں۔ اگر آپ کمیونٹی کے ساتھ اپنا ماڈل اشتراک کرنا چاہتے ہیں، تو `push_to_hub` کو `True` پر سیٹ کریں (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا چاہیے)۔ اس صورت میں `output_dir` ریپو کا نام بھی ہو گا جہاں آپ کے ماڈل چیک پوائنٹ کو آگے بڑھایا جائے گا۔

```py
>>> from transformers import TrainingArguments

>>> # REPLACE THIS WITH YOUR REPO ID
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     eval_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

مثالوں کو ایک ساتھ بیچنے کے لیے ایک سادہ ڈیٹا کولیٹر کی وضاحت کریں۔

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

آخر میں، سب کچھ ساتھ لائیں، اور کال کریں [`~Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     processing_class=processor,
... )

>>> trainer.train()
```

حتمی ماڈل کو 🤗 Hub میں شامل کرنے کے لیے، ماڈل کارڈ بنائیں اور `push_to_hub` کو کال کریں:

```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## انفرنس

اب جب کہ آپ نے ایک LayoutLMv2 ماڈل کو ٹھیک کر لیا ہے، اور اسے 🤗 Hub پر اپ لوڈ کر دیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں۔ انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`Pipeline`] میں استعمال کریں۔

آئیے ایک مثال لیتے ہیں:

```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

اس کے بعد، اپنے ماڈل کے ساتھ دستاویز کے سوال کے جواب کے لیے ایک پائپ لائن کو فوری بنائیں، اور اس میں تصویر + سوال کا مجموعہ پاس کریں۔

```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

اگر آپ چاہیں تو آپ پائپ لائن کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں:

1. ایک تصویر اور سوال لیں، انہیں اپنے ماڈل سے پروسیسر کا استعمال کرتے ہوئے ماڈل کے لیے تیار کریں۔
2. نتیجہ آگے بھیجیں یا ماڈل کے ذریعے پری پروسیسنگ کریں۔
3. ماڈل `start_logits` اور `end_logits` لوٹاتا ہے، جو بتاتا ہے کہ کون سا ٹوکن جواب کے شروع میں ہے اور
کون سا ٹوکن جواب کے آخر میں ہے۔ دونوں کی شکل ہے (بیچ_سائز، ترتیب_لمبائی)۔
4. پیشین گوئی `start_idx` اور `end_idx` حاصل کرنے کے لیے `start_logits` اور `end_logits` دونوں کی آخری جہت پر ایک آرگمیکس لیں۔
5. جواب کو ٹوکنائزر کے ساتھ ڈی کوڈ کریں۔

```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```
