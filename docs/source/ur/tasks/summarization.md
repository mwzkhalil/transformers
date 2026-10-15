<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# خلاصہ

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

خلاصہ ایک دستاویز یا مضمون کا ایک چھوٹا ورژن بناتا ہے جو تمام اہم معلومات کو حاصل کرتا ہے۔ ترجمے کے ساتھ ساتھ، یہ ایک کام کی ایک اور مثال ہے جسے ترتیب بہ ترتیب کام کے طور پر وضع کیا جا سکتا ہے۔ خلاصہ یہ ہو سکتا ہے:

- Extractive: کسی دستاویز سے سب سے زیادہ متعلقہ معلومات نکالیں۔
- خلاصہ: نیا متن تخلیق کریں جو سب سے زیادہ متعلقہ معلومات کو حاصل کرے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. خلاصہ خلاصہ کے لیے [BillSum](https://huggingface.co/google-t5/t5-small) ڈیٹاسیٹ کے کیلیفورنیا اسٹیٹ بل سب سیٹ پر T5](https://huggingface.co/datasets/FiscalNote/billsum) Finetune۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/summarization) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate rouge_score
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## بل سم ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے بل سم ڈیٹاسیٹ کے چھوٹے کیلیفورنیا اسٹیٹ بل سبسیٹ کو لوڈ کرکے شروع کریں:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("FiscalNote/billsum", split="ca_test")
```

ڈیٹاسیٹ کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ سے ٹیسٹ سیٹ کریں:

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

دو فیلڈز ہیں جو آپ استعمال کرنا چاہیں گے:

- `text`: بل کا متن جو ماڈل کا ان پٹ ہوگا۔
- `summary`: `text` کا گاڑھا ورژن جو ماڈل ہدف ہو گا۔

## پری پروسیس

اگلا مرحلہ `text` اور `summary` پر کارروائی کرنے کے لیے T5 ٹوکنائزر لوڈ کرنا ہے:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

پری پروسیسنگ فنکشن جس کو آپ بنانا چاہتے ہیں اس کی ضرورت ہے:

1. ان پٹ کو پرامپٹ کے ساتھ پریفکس کریں تاکہ T5 کو معلوم ہو کہ یہ خلاصہ کرنے کا کام ہے۔ کچھ ماڈلز ایک سے زیادہ NLP کاموں کے لیے مخصوص کاموں کے لیے اشارہ کرنے کی ضرورت ہوتی ہے۔
2. لیبلز کو ٹوکنائز کرتے وقت کلیدی لفظ `text_target` دلیل استعمال کریں۔
3. `max_length` پیرامیٹر کے ذریعے ترتیب دی گئی زیادہ سے زیادہ لمبائی سے زیادہ لمبا نہ ہونے کے لیے ترتیب کو چھوٹا کریں۔

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] طریقہ استعمال کریں۔ آپ ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کر کے `map` فنکشن کو تیز کر سکتے ہیں:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

اب [`DataCollatorForSeq2Seq`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ مجموعہ کے دوران تمام ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے جملوں کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) میٹرک لوڈ کریں (میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے طریقے کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں):

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

پھر ایک فنکشن بنائیں جو ROUGE میٹرک کا حساب لگانے کے لیے آپ کی پیشین گوئیوں اور لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForSeq2SeqLM`] کے ساتھ T5 لوڈ کریں:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`Seq2SeqTrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] ROUGE میٹرک کی جانچ کرے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Seq2SeqTrainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

خلاصہ کے لیے ماڈل کو ٹھیک کرنے کے طریقے کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

کچھ متن کے ساتھ آئیں جس کا خلاصہ آپ کرنا چاہتے ہیں۔ T5 کے لیے، آپ جس کام پر کام کر رہے ہیں اس کے لحاظ سے آپ کو اپنے ان پٹ کا سابقہ ​​لگانے کی ضرورت ہے۔ خلاصہ کے لیے آپ کو اپنے ان پٹ کا سابقہ ​​لگانا چاہیے جیسا کہ ذیل میں دکھایا گیا ہے:

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

متن کو ٹوکنائز کریں اور `input_ids` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

خلاصہ بنانے کے لیے [`~generation.GenerationMixin.generate`] طریقہ استعمال کریں۔ متن بنانے کی مختلف حکمت عملیوں اور جنریشن کو کنٹرول کرنے کے پیرامیٹرز کے بارے میں مزید تفصیلات کے لیے، [Text Generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) API کو دیکھیں۔

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

تیار کردہ ٹوکن آئی ڈی کو واپس متن میں ڈی کوڈ کریں:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
