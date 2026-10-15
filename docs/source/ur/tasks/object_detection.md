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

# آبجیکٹ کا پتہ لگانا

[[open-in-colab]]

آبجیکٹ کا پتہ لگانا ایک تصویر میں مثالوں (جیسے انسانوں، عمارتوں یا کاروں) کا پتہ لگانے کا کمپیوٹر وژن کا کام ہے۔ آبجیکٹ کا پتہ لگانا ماڈلز باؤنڈنگ خانوں کے ان پٹ اور آؤٹ پٹ کوآرڈینیٹ کے طور پر ایک تصویر وصول کرتا ہے اور پتہ چلنے والی اشیاء کے متعلقہ لیبلز۔ ایک تصویر متعدد اشیاء پر مشتمل ہو سکتی ہے، ہر ایک کا اپنا باؤنڈنگ باکس اور ایک لیبل (جیسے اس میں ایک کار اور ایک عمارت ہو سکتی ہے)، اور ہر شے تصویر کے مختلف حصوں میں موجود ہو سکتی ہے (مثلاً تصویر میں کئی کاریں ہو سکتی ہیں)۔ یہ کام عام طور پر خود مختار ڈرائیونگ میں پیدل چلنے والوں، سڑک کے نشانات اور ٹریفک لائٹس جیسی چیزوں کا پتہ لگانے کے لیے استعمال ہوتا ہے۔ دیگر ایپلی کیشنز میں تصاویر میں اشیاء کی گنتی، تصویر کی تلاش، اور بہت کچھ شامل ہے۔

اس گائیڈ میں، آپ سیکھیں گے کہ کیسے:

 1. فائن ٹیون [Mobile-ui-design](https://huggingface.co/Roboflow/rf-detr-medium) پر RF-DETR](https://huggingface.co/datasets/merve/mobile-ui-design)
موبائل ایپ کے اسکرین شاٹس میں UI عناصر کا پتہ لگانے کے لیے ڈیٹا سیٹ۔
 2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/object-detection) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install -q datasets transformers accelerate timm trackio torchmetrics pycocotools
```

آپ Hugging Face حب سے ڈیٹاسیٹ لوڈ کرنے کے لیے 🤗 ڈیٹا سیٹ اور اپنے ماڈل کو تربیت دینے کے لیے 🤗 Transformers استعمال کریں گے۔

ہم آپ کی حوصلہ افزائی کرتے ہیں کہ اپنا ماڈل کمیونٹی کے ساتھ شیئر کریں۔ اسے حب پر اپ لوڈ کرنے کے لیے اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

عالمی مستقل کی وضاحت کریں، یعنی ماڈل نام اور تصویر کا سائز۔ یہ ٹیوٹوریل RF-DETR کا استعمال کرتا ہے، لیکن آپ Transformers میں کسی بھی چیز کی شناخت ماڈل کو منتخب کر سکتے ہیں۔

```py
>>> MODEL_NAME = "Roboflow/rf-detr-medium"
```

## موبائل-UI-ڈیزائن ڈیٹاسیٹ لوڈ کریں۔

[mobile-ui-design dataset](https://huggingface.co/datasets/merve/mobile-ui-design) موبائل ایپ کے اسکرین شاٹس پر مشتمل ہے جس میں متن، تصاویر، مستطیل اور گروپس جیسے UI عناصر کا پتہ لگانے کے لیے تشریحات شامل ہیں۔

ڈیٹاسیٹ لوڈ کرکے اور زمرہ کے لیبل نکال کر شروع کریں۔ ڈیٹاسیٹ میں پہلے ہی اسپلٹس ہیں۔

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/mobile-ui-design")

>>> CATEGORIES = sorted(set(
...     cat for split in ds.values() for example in split for cat in example["objects"]["category"]
... ))
>>> label2id = {label: i for i, label in enumerate(CATEGORIES)}
>>> id2label = {i: label for label, i in label2id.items()}
>>> print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
Categories (4): ['group', 'image', 'rectangle', 'text']
```

ڈیٹاسیٹ COCO فارمیٹ `(x, y, w, h)` میں سٹرنگ زمرہ کے نام اور باؤنڈنگ بکس استعمال کرتا ہے۔ زمرہ جات کو انٹیجر آئی ڈیز، کمپیوٹ ایریاز میں تبدیل کریں، اور تربیت سے پہلے ڈیجنریٹ باؤنڈنگ بکس کو فلٹر کریں:

```py
>>> def prepare_example(example, idx):
...     objects = example["objects"]
...     bboxes = objects["bbox"]
...     categories = objects["category"]
...     img_w, img_h = example["width"], example["height"]
...     bboxes, cats, areas, ids = [], [], [], []
...     for i, (bbox, cat) in enumerate(zip(bboxes, categories)):
...         x, y, w, h = bbox
...         if w <= 0 or h <= 0:
...             continue
...         x = max(0.0, min(x, img_w))
...         y = max(0.0, min(y, img_h))
...         w = min(w, img_w - x)
...         h = min(h, img_h - y)
...         if w <= 0 or h <= 0:
...             continue
...         bboxes.append([x, y, w, h])
...         cats.append(label2id[cat])
...         areas.append(w * h)
...         ids.append(i)
...     return {
...         "image_id": idx, "image": example["image"],
...         "width": example["width"], "height": example["height"],
...         "objects": {"id": ids, "bbox": bboxes, "category": cats, "area": areas},
...     }

>>> ds_prepared = ds["train"].map(prepare_example, with_indices=True, remove_columns=ds["train"].column_names)
>>> ds_prepared = ds_prepared.filter(lambda x: len(x["objects"]["bbox"]) > 0)

>>> split = ds_prepared.train_test_split(test_size=0.15, seed=1337)
>>> train_ds = split["train"]
>>> val_ds = split["test"]
>>> print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")
Train: 6669, Validation: 1177
```

## ڈیٹا کو پہلے سے پروسیس کریں۔

[`AutoImageProcessor`] `pixel_values`، `pixel_mask`، اور `labels` بنانے کے لیے تصویری ڈیٹا کی پروسیسنگ کا خیال رکھتا ہے جس کے ساتھ ماڈل تربیت دے سکتا ہے۔ امیج پروسیسر نیا سائز، پیڈنگ، اور نارملائزیشن کو ہینڈل کرتا ہے۔ اس کے اوپری حصے میں، آپ اختیاری طور پر بے ترتیب ڈیٹا کو بڑھا سکتے ہیں (دیکھیں [نیچے](#data-augmentation)) کو بہتر بنانے کے لیے۔

```py
>>> import numpy as np
>>> from functools import partial
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
```

`image_processor` COCO فارمیٹ میں تشریحات کی توقع کرتا ہے: `{'image_id': int, 'annotations': list[Dict]}`۔ ہر مثال کی تشریحات کو فارمیٹ کریں اور پروسیسر کو باقی کو سنبھالنے دیں:

```py
>>> def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
...     annotations = []
...     for category, area, bbox in zip(categories, areas, bboxes):
...         annotations.append({
...             "image_id": image_id,
...             "category_id": category,
...             "iscrowd": 0,
...             "area": area,
...             "bbox": list(bbox),
...         })
...     return {"image_id": image_id, "annotations": annotations}

>>> def transform_batch(examples, image_processor):
...     images = []
...     annotations = []
...     for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
...         images.append(np.array(image.convert("RGB")))
...         formatted = format_image_annotations_as_coco(
...             image_id, objects["category"], objects["area"], objects["bbox"]
...         )
...         annotations.append(formatted)
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")
...     result.pop("pixel_mask", None)
...     return result

>>> transform_fn = partial(transform_batch, image_processor=image_processor)
>>> train_ds = train_ds.with_transform(transform_fn)
>>> val_ds = val_ds.with_transform(transform_fn)
```

<a id='data-augmentation'></a>
### ڈیٹا میں اضافہ

اوپر کی تبدیلی صرف تصاویر کا سائز تبدیل کرتی ہے اور اسے معمول بناتی ہے۔ **ٹریننگ** اسپلٹ پر لاگو بے ترتیب اضافہ عام طور پر عامیت کو بہتر بناتا ہے، جب کہ توثیق کی تقسیم کو اضافہ سے پاک رہنا چاہیے تاکہ تشخیص تعییناتی رہے۔ ایک عام انتخاب ہے [Albumentations](https://albumentations.ai/)، جو تصویر اور اس کے باؤنڈنگ بکس کو ایک ساتھ بڑھاتا ہے۔ `bbox_params` کے ساتھ پائپ لائن کی وضاحت کریں تاکہ خانوں کو تصویر کے ساتھ مستقل طور پر تبدیل کیا جائے، پھر بڑھے ہوئے خانوں سے علاقوں کی دوبارہ گنتی کریں:

```py
>>> import albumentations as A

>>> train_augment = A.Compose(
...     [
...         A.Perspective(p=0.1),
...         A.HorizontalFlip(p=0.5),
...         A.RandomBrightnessContrast(p=0.5),
...         A.HueSaturationValue(p=0.1),
...     ],
...     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
... )

>>> def augment_and_transform_batch(examples, image_processor, transform):
...     images = []
...     annotations = []
...     for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
...         image = np.array(image.convert("RGB"))
...         output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
...         images.append(output["image"])
...         areas = [w * h for (_, _, w, h) in output["bboxes"]]
...         formatted = format_image_annotations_as_coco(
...             image_id, output["category"], areas, output["bboxes"]
...         )
...         annotations.append(formatted)
...     result = image_processor(images=images, annotations=annotations, return_tensors="pt")
...     result.pop("pixel_mask", None)
...     return result
```

صرف ٹریننگ اسپلٹ پر اضافہ کرنے والی تبدیلی کا اطلاق کریں، اور توثیق کے لیے سادہ `transform_fn` رکھیں:

```py
>>> train_augment_fn = partial(augment_and_transform_batch, image_processor=image_processor, transform=train_augment)
>>> train_ds = train_ds.with_transform(train_augment_fn)
>>> val_ds = val_ds.with_transform(transform_fn)
```

تصاویر کو ایک ساتھ بیچنے کے لیے حسب ضرورت `collate_fn` بنائیں:

```py
>>> import torch

>>> def collate_fn(batch):
...     data = {}
...     data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
...     data["labels"] = [x["labels"] for x in batch]
...     if "pixel_mask" in batch[0]:
...         data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
...     return data

```

## ایم اے پی کی گنتی کے لیے فنکشن کی تیاری

آبجیکٹ کا پتہ لگاناماڈلزکے ایک سیٹ کے ساتھ عام طور پر جانچا جاتا ہے۔<a href="https://cocodataset.org/#detection-eval">COCO طرز کے میٹرکس</a>. ہم استعمال کرنے جا رہے ہیں`torchmetrics`حساب کرنا`mAP`(یعنی اوسط صحت سے متعلق) اور`mAR`(مطلب اوسط یاد) میٹرکس اور اسے لپیٹ دیں گے۔`compute_metrics`میں استعمال کرنے کے لیے فنکشن[`Trainer`]تشخیص کے لیے

ٹریننگ کے لیے استعمال کیے جانے والے بکسوں کا انٹرمیڈیٹ فارمیٹ `YOLO` (نارملائزڈ) ہے لیکن باکس ایریاز کو صحیح طریقے سے سنبھالنے کے لیے ہم `Pascal VOC` (مطلق) فارمیٹ میں بکس کے لیے میٹرکس کی گنتی کریں گے۔ آئیے ایک فنکشن کی وضاحت کرتے ہیں جو باؤنڈنگ بکس کو `Pascal VOC` فارمیٹ میں تبدیل کرتا ہے:

```py
>>> from transformers.image_transforms import center_to_corners_format

>>> def convert_bbox_yolo_to_pascal(boxes, image_size):
...     """
...     Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
...     to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

...     Args:
...         boxes (torch.Tensor): Bounding boxes in YOLO format
...         image_size (tuple[int, int]): Image size in format (height, width)

...     Returns:
...         torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
...     """
...     # convert center to corners format
...     boxes = center_to_corners_format(boxes)

...     # convert to absolute coordinates
...     height, width = image_size
...     boxes = boxes * torch.tensor([[width, height, width, height]])

...     return boxes
```

پھر، `compute_metrics` فنکشن میں ہم `predicted` اور `target` باؤنڈنگ باکسز، اسکورز اور لیبلز کو تشخیصی لوپ کے نتائج سے جمع کرتے ہیں اور اسے اسکورنگ فنکشن میں منتقل کرتے ہیں۔

```py
>>> import numpy as np
>>> from dataclasses import dataclass
>>> from torchmetrics.detection.mean_ap import MeanAveragePrecision


>>> @dataclass
>>> class ModelOutput:
...     logits: torch.Tensor
...     pred_boxes: torch.Tensor


>>> def _get_orig_size(image_target):
...     """Robust orig_size extraction - Trainer serialization can truncate to 1 element."""
...     orig = np.atleast_1d(np.asarray(image_target["orig_size"])).flatten()
...     if len(orig) >= 2:
...         return (int(orig[0]), int(orig[1]))
...     return (int(orig[0]), int(orig[0]))

>>> @torch.no_grad()
>>> def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
...     predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
...     image_sizes = []
...     post_processed_targets = []
...     post_processed_predictions = []
...
...     for batch in targets:
...         batch_sizes = []
...         for image_target in batch:
...             h, w = _get_orig_size(image_target)
...             batch_sizes.append([h, w])
...             boxes = torch.tensor(image_target["boxes"])
...             boxes = convert_bbox_yolo_to_pascal(boxes, (h, w))
...             labels = torch.tensor(image_target["class_labels"])
...             post_processed_targets.append({"boxes": boxes, "labels": labels})
...         image_sizes.append(torch.tensor(batch_sizes))
...
...     for batch, target_sizes in zip(predictions, image_sizes):
...         batch_logits, batch_boxes = batch[1], batch[2]
...         output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
...         post_processed_output = image_processor.post_process_object_detection(
...             output, threshold=threshold, target_sizes=target_sizes
...         )
...         post_processed_predictions.extend(post_processed_output)
...
...     metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
...     metric.update(post_processed_predictions, post_processed_targets)
...     metrics = metric.compute()
...
...     classes = metrics.pop("classes")
...     map_per_class = metrics.pop("map_per_class")
...     mar_100_per_class = metrics.pop("mar_100_per_class")
...     for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
...         class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
...         metrics[f"map_{class_name}"] = class_map
...         metrics[f"mar_100_{class_name}"] = class_mar
...
...     metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
...     return metrics

>>> eval_compute_metrics_fn = partial(
...     compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
... )
```

## پتہ لگانے کی تربیت ماڈل

آپ نے پچھلے حصوں میں زیادہ تر بھاری لفٹنگ کی ہے، لہذا اب آپ اپنے ماڈل کو تربیت دینے کے لیے تیار ہیں! سائز تبدیل کرنے کے بعد بھی اس ڈیٹاسیٹ میں تصاویر اب بھی کافی بڑی ہیں۔ اس کا مطلب ہے کہ اس ماڈل کو ٹھیک کرنے کے لیے کم از کم ایک GPU کی ضرورت ہوگی۔

تربیت میں درج ذیل مراحل شامل ہیں:

1. ماڈل کو [`AutoModelForObjectDetection`] کے ساتھ اسی چیک پوائنٹ کا استعمال کرتے ہوئے لوڈ کریں جیسا کہ پری پروسیسنگ میں ہے۔
2. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔
3. ماڈل، ڈیٹاسیٹ، امیج پروسیسر، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
4. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

اسی چیک پوائنٹ سے ماڈل لوڈ کرتے وقت جو آپ نے پری پروسیسنگ کے لیے استعمال کیا تھا، یاد رکھیں کہ `label2id` اور `id2label` نقشے جو آپ نے پہلے ڈیٹا سیٹ کے میٹا ڈیٹا سے بنائے تھے۔ مزید برآں، ہم `ignore_mismatched_sizes=True` کی وضاحت کرتے ہیں تاکہ موجودہ درجہ بندی ہیڈ کو نئے سرے سے تبدیل کیا جا سکے۔

```py
>>> from transformers import AutoModelForObjectDetection

>>> model = AutoModelForObjectDetection.from_pretrained(
...     MODEL_NAME,
...     id2label=id2label,
...     label2id=label2id,
...     ignore_mismatched_sizes=True,
... )
```

[`TrainingArguments`] میں یہ بتانے کے لیے `output_dir` استعمال کریں کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے، پھر جیسا کہ آپ مناسب دیکھیں ہائپر پیرامیٹر کو ترتیب دیں۔ `num_train_epochs=5` ٹریننگ کے لیے A100 GPU پر تقریباً 35 منٹ لگیں گے، بہتر نتائج حاصل کرنے کے لیے عہدوں کی تعداد میں اضافہ کریں۔

اہم نوٹس:

- غیر استعمال شدہ کالم نہ ہٹائیں کیونکہ اس سے تصویری کالم گر جائے گا۔ تصویری کالم کے بغیر، آپ
`pixel_values` نہیں بنا سکتے۔ اس وجہ سے، `remove_unused_columns` کو `False` پر سیٹ کریں۔
- درست تشخیص کے نتائج حاصل کرنے کے لیے `eval_do_concat_batches=False` سیٹ کریں۔ امیجز میں ٹارگٹ باکسز کی مختلف تعداد ہوتی ہے، اگر بیچوں کو جوڑ دیا جاتا ہے تو ہم اس بات کا تعین نہیں کر پائیں گے کہ کون سے بکس خاص تصویر سے تعلق رکھتے ہیں۔

اگر آپ حب پر دھکیل کر اپنا ماڈل شیئر کرنا چاہتے ہیں تو `push_to_hub` کو `True` پر سیٹ کریں (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا چاہیے)۔

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="rf_detr_finetuned_mobile_ui",
...     num_train_epochs=5,
...     bf16=True,
...     per_device_train_batch_size=8,
...     dataloader_num_workers=4,
...     learning_rate=5e-5,
...     lr_scheduler_type="cosine",
...     weight_decay=1e-4,
...     max_grad_norm=0.01,
...     metric_for_best_model="eval_map",
...     greater_is_better=True,
...     load_best_model_at_end=True,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     save_total_limit=2,
...     remove_unused_columns=False,
...     report_to="trackio",
...     run_name="mobile-ui-detection",
...     eval_do_concat_batches=False,
...     push_to_hub=True,
... )
```

آخر میں، سب کچھ ساتھ لائیں، اور کال کریں [`~transformers.Trainer.train`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=val_ds,
...     processing_class=image_processor,
...     data_collator=collate_fn,
...     compute_metrics=eval_compute_metrics_fn,
... )

>>> trainer.train()
```

<div>

  <progress value='2085' max='2085' style='width:300px; height:20px; vertical-align: middle;'></progress>
[2085/2085 38:39، عہد 5/5]
</div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Map</th>
      <th>Map 50</th>
      <th>Map 75</th>
      <th>Map Small</th>
      <th>Map Medium</th>
      <th>Map Large</th>
      <th>Mar 1</th>
      <th>Mar 10</th>
      <th>Mar 100</th>
      <th>Mar Small</th>
      <th>Mar Medium</th>
      <th>Mar Large</th>
      <th>Map Group</th>
      <th>Mar 100 Group</th>
      <th>Map Image</th>
      <th>Mar 100 Image</th>
      <th>Map Rectangle</th>
      <th>Mar 100 Rectangle</th>
      <th>Map Text</th>
      <th>Mar 100 Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>9.9234</td>
      <td>0.1303</td>
      <td>0.2236</td>
      <td>0.1478</td>
      <td>0.0909</td>
      <td>0.2030</td>
      <td>0.2524</td>
      <td>0.0421</td>
      <td>0.2520</td>
      <td>0.4683</td>
      <td>0.3113</td>
      <td>0.5607</td>
      <td>0.6782</td>
      <td>0.1244</td>
      <td>0.5122</td>
      <td>0.0958</td>
      <td>0.5035</td>
      <td>0.1285</td>
      <td>0.4328</td>
      <td>0.1725</td>
      <td>0.4413</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>9.8472</td>
      <td>0.1893</td>
      <td>0.3017</td>
      <td>0.2124</td>
      <td>0.1347</td>
      <td>0.2789</td>
      <td>0.3038</td>
      <td>0.0549</td>
      <td>0.2961</td>
      <td>0.5140</td>
      <td>0.3433</td>
      <td>0.5941</td>
      <td>0.7406</td>
      <td>0.1305</td>
      <td>0.5423</td>
      <td>0.1979</td>
      <td>0.5578</td>
      <td>0.1964</td>
      <td>0.4648</td>
      <td>0.2324</td>
      <td>0.4437</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>9.6401</td>
      <td>0.2275</td>
      <td>0.3547</td>
      <td>0.2657</td>
      <td>0.1698</td>
      <td>0.3336</td>
      <td>0.3892</td>
      <td>0.0611</td>
      <td>0.3204</td>
      <td>0.5270</td>
      <td>0.3625</td>
      <td>0.6143</td>
      <td>0.7496</td>
      <td>0.1602</td>
      <td>0.5684</td>
      <td>0.2617</td>
      <td>0.5763</td>
      <td>0.2249</td>
      <td>0.4684</td>
      <td>0.2631</td>
      <td>0.4692</td>
    </tr>
    <tr>
      <td>4</td>
      <td>No log</td>
      <td>9.5770</td>
      <td>0.2733</td>
      <td>0.4068</td>
      <td>0.3133</td>
      <td>0.2100</td>
      <td>0.3867</td>
      <td>0.4343</td>
      <td>0.0668</td>
      <td>0.3456</td>
      <td>0.5593</td>
      <td>0.3875</td>
      <td>0.6393</td>
      <td>0.7725</td>
      <td>0.2013</td>
      <td>0.5941</td>
      <td>0.3158</td>
      <td>0.6065</td>
      <td>0.2733</td>
      <td>0.4998</td>
      <td>0.3028</td>
      <td>0.4756</td>
    </tr>
    <tr>
      <td>5</td>
      <td>10.3700</td>
      <td>11.0500</td>
      <td>0.2827</td>
      <td>0.4193</td>
      <td>0.2913</td>
      <td>0.2021</td>
      <td>0.2814</td>
      <td>0.3763</td>
      <td>0.0609</td>
      <td>0.3403</td>
      <td>0.5668</td>
      <td>0.4138</td>
      <td>0.5669</td>
      <td>0.7317</td>
      <td>0.2092</td>
      <td>0.5979</td>
      <td>0.3334</td>
      <td>0.6295</td>
      <td>0.2793</td>
      <td>0.5245</td>
      <td>0.3089</td>
      <td>0.5151</td>
    </tr>
  </tbody>
</table><p>

اگر آپ نے `training_args` میں `push_to_hub` کو `True` پر سیٹ کیا ہے، تو ٹریننگ چیک پوائنٹس کو Hugging Face حب کی طرف دھکیل دیا جائے گا۔ تربیت مکمل ہونے پر، [`~transformers.Trainer.push_to_hub`] طریقہ پر کال کرکے حتمی ماڈل کو حب کی طرف دھکیلیں۔

```py
>>> trainer.push_to_hub()
```

## اندازہ لگانا

```py
>>> from pprint import pprint

>>> metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="test")
>>> pprint(metrics)
{'test_loss': 11.05,
 'test_map': 0.2827,
 'test_map_50': 0.4193,
 'test_map_75': 0.2913,
 'test_map_group': 0.2092,
 'test_map_image': 0.3334,
 'test_map_large': 0.3763,
 'test_map_medium': 0.2814,
 'test_map_rectangle': 0.2793,
 'test_map_small': 0.2021,
 'test_map_text': 0.3089,
 'test_mar_1': 0.0609,
 'test_mar_10': 0.3403,
 'test_mar_100': 0.5668,
 'test_mar_100_group': 0.5979,
 'test_mar_100_image': 0.6295,
 'test_mar_100_rectangle': 0.5245,
 'test_mar_100_text': 0.5151,
 'test_mar_large': 0.7317,
 'test_mar_medium': 0.5669,
 'test_mar_small': 0.4138}
```

ان نتائج کو عہدوں کی تعداد میں اضافہ کرکے یا [`TrainingArguments`] میں دوسرے ہائپر پیرامیٹر کو ایڈجسٹ کرکے مزید بہتر کیا جا سکتا ہے۔ اسے جانے دو!

## انفرنس

اب جب کہ آپ نے ایک ماڈل کو ٹھیک کر لیا ہے، اس کا اندازہ کیا ہے، اور اسے Hugging Face Hub پر اپ لوڈ کر دیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں۔

```py
>>> import torch
>>> from PIL import Image, ImageDraw
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/mobile-ui-design", split="train")
>>> image = ds[5]["image"].convert("RGB")
```

ماڈل اور امیج پروسیسر کو Hugging Face حب سے لوڈ کریں (اس سیشن میں پہلے سے تربیت یافتہ استعمال کرنا چھوڑ دیں):

```py
>>> model_repo = "merve/rf_detr_finetuned_mobile_ui"

>>> image_processor = AutoImageProcessor.from_pretrained(model_repo)
>>> model = AutoModelForObjectDetection.from_pretrained(model_repo)
>>> model.eval()
```

اور باؤنڈنگ بکس کا پتہ لگائیں:

```py
>>> with torch.no_grad():
...     inputs = image_processor(images=[image], return_tensors="pt")
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([[image.size[1], image.size[0]]])
...     results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected text with confidence 0.727 at location [324.02, 340.55, 339.52, 359.12]
Detected rectangle with confidence 0.717 at location [39.97, 705.14, 335.93, 753.54]
Detected text with confidence 0.702 at location [199.94, 473.66, 213.41, 490.6]
Detected text with confidence 0.678 at location [153.14, 474.81, 165.33, 491.0]
Detected text with confidence 0.675 at location [262.67, 718.28, 281.44, 740.81]
Detected rectangle with confidence 0.655 at location [143.57, 242.51, 214.32, 274.26]
Detected text with confidence 0.653 at location [298.68, 637.77, 345.68, 656.26]
```

آئیے نتیجہ پلاٹ کریں:

```py
>>> draw = ImageDraw.Draw(image)

>>> colors = {"group": "blue", "image": "green", "rectangle": "red", "text": "orange"}
>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     x, y, x2, y2 = tuple(box)
...     label_name = model.config.id2label[label.item()]
...     color = colors.get(label_name, "red")
...     draw.rectangle((x, y, x2, y2), outline=color, width=2)
...     draw.text((x, y), f"{label_name} {score:.2f}", fill=color)

>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/mobile_ui_result_5.png" alt="Object detection result on a cart screen"/>
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/mobile_ui_result_50.png" alt="Object detection result on a followers screen"/>
</div>
