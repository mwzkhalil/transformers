<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ویڈیو کی درجہ بندی

[[open-in-colab]]

ویڈیو کی درجہ بندی ایک پوری ویڈیو کو ایک لیبل یا کلاس تفویض کرنے کا کام ہے۔ ویڈیوز میں ہر ویڈیو کے لیے صرف ایک کلاس کی توقع کی جاتی ہے۔ ویڈیو کی درجہ بندی ماڈلز ایک ویڈیو کو بطور ان پٹ لیں اور پیشین گوئی واپس کریں کہ ویڈیو کس کلاس سے تعلق رکھتی ہے۔ ان ماڈلز کو اس بات کی درجہ بندی کرنے کے لیے استعمال کیا جا سکتا ہے کہ ویڈیو کیا ہے۔ ویڈیو کی درجہ بندی کی ایک حقیقی دنیا کی ایپلی کیشن ایکشن/سرگرمی کی شناخت ہے، جو فٹنس ایپلی کیشنز کے لیے مفید ہے۔ یہ بصارت سے محروم افراد کے لیے بھی مددگار ہے، خاص طور پر جب وہ سفر کر رہے ہوں۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) ڈیٹاسیٹ کے ذیلی سیٹ پر۔
2. انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس اس ٹاسک کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/video-classification) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install -q pytorchvideo transformers evaluate
```

آپ ویڈیوز کو پروسیس کرنے اور تیار کرنے کے لیے [PyTorchVideo](https://pytorchvideo.org/) (ڈب شدہ `pytorchvideo`) استعمال کریں گے۔

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## UCF101 ڈیٹاسیٹ لوڈ کریں۔

[UCF-101 ڈیٹا سیٹ](https://www.crcv.ucf.edu/data/UCF101.php) کا سب سیٹ لوڈ کر کے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

سب سیٹ ڈاؤن لوڈ ہونے کے بعد، آپ کو کمپریسڈ آرکائیو کو نکالنے کی ضرورت ہے:

```py
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...      t.extractall(".")
```

اعلی سطح پر، ڈیٹاسیٹ کو اس طرح ترتیب دیا جاتا ہے:

```bash
UCF101_subset/
    train/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    val/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    test/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
```

اس کے بعد آپ کل ویڈیوز کی تعداد گن سکتے ہیں۔

```py
>>> import pathlib
>>> dataset_root_path = "UCF101_subset"
>>> dataset_root_path = pathlib.Path(dataset_root_path)
```

```py
>>> video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
>>> video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
>>> video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
>>> video_total = video_count_train + video_count_val + video_count_test
>>> print(f"Total videos: {video_total}")
```

```py
>>> all_video_file_paths = (
...     list(dataset_root_path.glob("train/*/*.avi"))
...     + list(dataset_root_path.glob("val/*/*.avi"))
...     + list(dataset_root_path.glob("test/*/*.avi"))
...  )
>>> all_video_file_paths[:5]
```

(`sorted`) ویڈیو کے راستے اس طرح ظاہر ہوتے ہیں:

```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

آپ دیکھیں گے کہ اسی گروپ/منظر سے تعلق رکھنے والے ویڈیو کلپس موجود ہیں جہاں ویڈیو فائل پاتھز میں گروپ کو `g` سے ظاہر کیا گیا ہے۔ `v_ApplyEyeMakeup_g07_c04.avi` اور `v_ApplyEyeMakeup_g07_c06.avi`، مثال کے طور پر۔

توثیق اور تشخیص کی تقسیم کے لیے، آپ [ڈیٹا لیکیج](https://www.kaggle.com/code/alexisbcook/data-leakage) کو روکنے کے لیے ایک ہی گروپ/منظر سے ویڈیو کلپس نہیں لینا چاہیں گے۔ اس ٹیوٹوریل میں آپ جو سب سیٹ استعمال کر رہے ہیں وہ اس معلومات کو مدنظر رکھتا ہے۔

اس کے بعد، آپ ڈیٹاسیٹ میں موجود لیبلز کا سیٹ حاصل کریں گے۔ نیز، دو لغتیں بنائیں جو ماڈل کو شروع کرتے وقت مددگار ثابت ہوں گی۔

* `label2id`: کلاس کے ناموں کو عدد میں نقشہ بناتا ہے۔
* `id2label`: انٹیجرز کو کلاس کے ناموں پر نقشہ بناتا ہے۔

```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# Unique classes: ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress'].
```

10 منفرد کلاسز ہیں۔ ہر کلاس کے لیے، تربیتی سیٹ میں 30 ویڈیوز ہیں۔

## ماڈل سے فائن ٹیون لوڈ کریں

ایک پہلے سے تربیت یافتہ چیک پوائنٹ اور اس سے وابستہ امیج پروسیسر سے ویڈیو کی درجہ بندی ماڈل فوری کریں۔ ماڈل کا انکوڈر پہلے سے تربیت یافتہ پیرامیٹرز کے ساتھ آتا ہے، اور درجہ بندی کا سر تصادفی طور پر شروع کیا جاتا ہے۔ ہمارے ڈیٹاسیٹ کے لیے پری پروسیسنگ پائپ لائن لکھتے وقت امیج پروسیسر کام آئے گا۔

```py
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
... )
```

جب ماڈل لوڈ ہو رہا ہے، تو آپ کو درج ذیل انتباہ نظر آ سکتا ہے:

```bash
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of VideoMAEForVideoClassification were not initialized from the model checkpoint at MCG-NJU/videomae-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

وارننگ ہمیں بتا رہی ہے کہ ہم کچھ وزن پھینک رہے ہیں (جیسے `classifier` پرت کا وزن اور تعصب) اور کچھ دوسرے کو تصادفی طور پر شروع کر رہے ہیں (ایک نئی `classifier` پرت کا وزن اور تعصب)۔ اس معاملے میں اس کی توقع کی جاتی ہے، کیونکہ ہم ایک نیا ہیڈ شامل کر رہے ہیں جس کے لیے ہمارے پاس پہلے سے تربیت یافتہ وزن نہیں ہے، اس لیے لائبریری ہمیں متنبہ کرتی ہے کہ ہمیں اسے انفرنس کے لیے استعمال کرنے سے پہلے اس فائن ٹیون کو ماڈل کرنا چاہیے، جو بالکل وہی ہے جو ہم کرنے جا رہے ہیں۔

**نوٹ** کہ [یہ چیک پوائنٹ](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) اس ٹاسک پر بہتر کارکردگی کا باعث بنتا ہے کیونکہ چیک پوائنٹ اسی طرح کے ڈاؤن اسٹریم ٹاسک پر فائن ٹیوننگ حاصل کیا گیا تھا جس میں کافی ڈومین اوورلیپ تھا۔ آپ چیک کر سکتے ہیں [یہ چیک پوائنٹ](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset) جو فائن ٹیوننگ `MCG-NJU/videomae-base-finetuned-kinetics` نے حاصل کیا تھا۔

## تربیت کے لیے ڈیٹاسیٹ تیار کریں۔

ویڈیوز کو پہلے سے پروسیس کرنے کے لیے، آپ [PyTorchVideo library](https://pytorchvideo.org/) کا فائدہ اٹھائیں گے۔ ان انحصارات کو درآمد کرکے شروع کریں جن کی ہمیں ضرورت ہے۔

```py
>>> import pytorchvideo.data

>>> from pytorchvideo.transforms import (
...     ApplyTransformToKey,
...     Normalize,
...     RandomShortSideScale,
...     RemoveKey,
...     ShortSideScale,
...     UniformTemporalSubsample,
... )

>>> from torchvision.transforms import (
...     Compose,
...     Lambda,
...     RandomCrop,
...     RandomHorizontalFlip,
...     Resize,
... )
```

ٹریننگ ڈیٹاسیٹ کی تبدیلیوں کے لیے، یکساں عارضی سب سیمپلنگ، پکسل نارملائزیشن، رینڈم کراپنگ، اور بے ترتیب افقی فلپنگ کا مجموعہ استعمال کریں۔ ڈیٹا سیٹ کی توثیق اور تشخیص کے لیے، بے ترتیب کراپنگ اور افقی فلپنگ کے علاوہ ایک ہی ٹرانسفارمیشن چین کو رکھیں۔ ان تبدیلیوں کی تفصیلات کے بارے میں مزید جاننے کے لیے [PyTorchVideo](https://pytorchvideo.org) کی سرکاری دستاویزات دیکھیں۔

درج ذیل معلومات حاصل کرنے کے لیے پہلے سے تربیت یافتہ ماڈل سے وابستہ `image_processor` استعمال کریں:

* تصویر کا مطلب اور معیاری انحراف جس کے ساتھ ویڈیو فریم پکسلز کو نارمل کیا جائے گا۔
* مقامی ریزولیوشن جس میں ویڈیو فریمز کا سائز تبدیل کیا جائے گا۔

کچھ مستقل کی وضاحت کرکے شروع کریں۔

```py
>>> mean = image_processor.image_mean
>>> std = image_processor.image_std
>>> if "shortest_edge" in image_processor.size:
...     height = width = image_processor.size["shortest_edge"]
>>> else:
...     height = image_processor.size["height"]
...     width = image_processor.size["width"]
>>> resize_to = (height, width)

>>> num_frames_to_sample = model.config.num_frames
>>> sample_rate = 4
>>> fps = 30
>>> clip_duration = num_frames_to_sample * sample_rate / fps
```

اب، ڈیٹاسیٹ سے متعلق مخصوص تبدیلیوں اور ڈیٹاسیٹس کی بالترتیب وضاحت کریں۔ تربیتی سیٹ کے ساتھ شروع کرنا:

```py
>>> train_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     RandomShortSideScale(min_size=256, max_size=320),
...                     RandomCrop(resize_to),
...                     RandomHorizontalFlip(p=0.5),
...                 ]
...             ),
...         ),
...     ]
... )

>>> train_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "train"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
...     decode_audio=False,
...     transform=train_transform,
... )
```

ورک فلو کی اسی ترتیب کو توثیق اور تشخیص کے سیٹ پر لاگو کیا جا سکتا ہے:

```py
>>> val_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     Resize(resize_to),
...                 ]
...             ),
...         ),
...     ]
... )

>>> val_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "val"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )

>>> test_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "test"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )
```

**نوٹ**: مندرجہ بالا ڈیٹاسیٹ پائپ لائنز [آفیشل PyTorchVideo example](https://pytorchvideo.org/docs/tutorial_classification#dataset) سے لی گئی ہیں۔ ہم [`pytorchvideo.data.Ucf101()`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.Ucf101) فنکشن استعمال کر رہے ہیں کیونکہ یہ UCF-101 ڈیٹاسیٹ کے لیے تیار کیا گیا ہے۔ ہڈ کے نیچے، یہ ایک [`pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.LabeledVideoDataset) آبجیکٹ واپس کرتا ہے۔ `LabeledVideoDataset` کلاس PyTorchVideo ڈیٹا سیٹ میں تمام چیزوں کی ویڈیو کے لیے بنیادی کلاس ہے۔ اس لیے، اگر آپ PyTorchVideo کے ذریعے آف دی شیلف میں تعاون یافتہ نہ ہونے والا حسب ضرورت ڈیٹا سیٹ استعمال کرنا چاہتے ہیں، تو آپ اس کے مطابق `LabeledVideoDataset` کلاس کو بڑھا سکتے ہیں۔ مزید جاننے کے لیے `data` API [دستاویزات](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html) سے رجوع کریں۔ اس کے علاوہ، اگر آپ کا ڈیٹا سیٹ اسی طرح کے ڈھانچے کی پیروی کرتا ہے (جیسا کہ اوپر دکھایا گیا ہے)، تو `pytorchvideo.data.Ucf101()` کا استعمال بالکل ٹھیک کام کرنا چاہیے۔

ڈیٹا سیٹ میں ویڈیوز کی تعداد جاننے کے لیے آپ `num_videos` دلیل تک رسائی حاصل کر سکتے ہیں۔

```py
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300, 30, 75)
```

## بہتر ڈیبگنگ کے لیے پہلے سے پروسیس شدہ ویڈیو کا تصور کریں۔

```py
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """Un-normalizes the image pixels."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """Prepares a GIF from a video tensor.
...
...     The video tensor is expected to have the following shape:
...     (num_frames, num_channels, height, width).
...     """
...     frames = []
...     for video_frame in video_tensor:
...         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
...         frames.append(frame_unnormalized)
...     kargs = {"duration": 0.25}
...     imageio.mimsave(filename, frames, "GIF", **kargs)
...     return filename

>>> def display_gif(video_tensor, gif_name="sample.gif"):
...     """Prepares and displays a GIF from a video tensor."""
...     video_tensor = video_tensor.permute(1, 0, 2, 3)
...     gif_filename = create_gif(video_tensor, gif_name)
...     return Image(filename=gif_filename)

>>> sample_video = next(iter(train_dataset))
>>> video_tensor = sample_video["video"]
>>> display_gif(video_tensor)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif.gif" alt="Person playing basketball"/>
</div>

## ماڈل کو تربیت دیں

ماڈل کو تربیت دینے کے لیے 🤗 Transformers سے [`Trainer`] کا فائدہ حاصل کریں۔ `Trainer` کو تیز کرنے کے لیے، آپ کو تربیت کی ترتیب اور تشخیصی میٹرک کی وضاحت کرنے کی ضرورت ہے۔ سب سے اہم [`TrainingArguments`] ہے، جو ایک کلاس ہے جس میں تربیت کو ترتیب دینے کے لیے تمام صفات شامل ہیں۔ اس کے لیے آؤٹ پٹ فولڈر کا نام درکار ہے، جو ماڈل کے چیک پوائنٹس کو محفوظ کرنے کے لیے استعمال کیا جائے گا۔ یہ 🤗 Hub پر ماڈل ریپوزٹری میں موجود تمام معلومات کو ہم آہنگ کرنے میں بھی مدد کرتا ہے۔

زیادہ تر تربیتی دلائل خود وضاحتی ہیں، لیکن ایک جو یہاں بہت اہم ہے وہ ہے `remove_unused_columns=False`۔ یہ کسی بھی خصوصیت کو چھوڑ دے گا جو ماڈل کے کال فنکشن میں استعمال نہیں کیا گیا ہے۔ پہلے سے طے شدہ طور پر یہ `True` ہے کیونکہ عام طور پر غیر استعمال شدہ فیچر کالمز کو چھوڑنا مثالی ہے، جس سے ماڈل کے کال فنکشن میں ان پٹ کو کھولنا آسان ہو جاتا ہے۔ لیکن، اس معاملے میں، آپ کو `pixel_values` بنانے کے لیے غیر استعمال شدہ خصوصیات (خاص طور پر 'ویڈیو') کی ضرورت ہے (جو کہ ایک لازمی کلید ہے جس کی ہماری ماڈل ان پٹس میں توقع کرتا ہے)۔

```py
>>> from transformers import TrainingArguments, Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name,
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=batch_size,
...     per_device_eval_batch_size=batch_size,
...     warmup_steps=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
... )
```

`pytorchvideo.data.Ucf101()` کے ذریعے لوٹایا گیا ڈیٹا سیٹ `__len__` طریقہ کو نافذ نہیں کرتا ہے۔ اس طرح، `TrainingArguments` کو انسٹینٹیٹ کرتے وقت ہمیں `max_steps` کی وضاحت کرنی چاہیے۔

اس کے بعد، آپ کو پیشین گوئیوں سے میٹرکس کی گنتی کرنے کے لیے ایک فنکشن کی وضاحت کرنے کی ضرورت ہے، جو `metric` استعمال کرے گا جسے آپ ابھی لوڈ کریں گے۔ آپ کو صرف پری پروسیسنگ کرنا ہے ہمارے پیشین گوئی شدہ لاگٹس کا آرگمیکس لینا ہے:

```py
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

**تشخیص پر ایک نوٹ**:

[VideoMAE پیپر](https://huggingface.co/papers/2203.12602) میں، مصنفین درج ذیل تشخیصی حکمت عملی استعمال کرتے ہیں۔ وہ ٹیسٹ ویڈیوز سے کئی کلپس پر ماڈل کا جائزہ لیتے ہیں اور ان کلپس پر مختلف فصلیں لگاتے ہیں اور مجموعی اسکور کی اطلاع دیتے ہیں۔ تاہم، سادگی اور اختصار کے مفاد میں، ہم اس ٹیوٹوریل میں اس پر غور نہیں کرتے۔

نیز، ایک `collate_fn` کی وضاحت کریں، جو مثالوں کو ایک ساتھ جمع کرنے کے لیے استعمال کیا جائے گا۔ ہر بیچ 2 کلیدوں پر مشتمل ہوتا ہے، یعنی `pixel_values` اور `labels`۔

```py
>>> def collate_fn(examples):
...     # permute to (num_frames, num_channels, height, width)
...     pixel_values = torch.stack(
...         [example["video"].permute(1, 0, 2, 3) for example in examples]
...     )
...     labels = torch.tensor([example["label"] for example in examples])
...     return {"pixel_values": pixel_values, "labels": labels}
```

پھر آپ یہ سب ڈیٹا سیٹس کے ساتھ `Trainer` پر منتقل کریں:

```py
>>> trainer = Trainer(
...     model,
...     args,
...     train_dataset=train_dataset,
...     eval_dataset=val_dataset,
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
...     data_collator=collate_fn,
... )
```

آپ حیران ہو سکتے ہیں کہ جب آپ نے پہلے ہی ڈیٹا کو پہلے سے پروسیس کیا تھا تو آپ `image_processor` کو ٹوکنائزر کے طور پر کیوں گزرے تھے۔ یہ صرف اس بات کو یقینی بنانے کے لیے ہے کہ امیج پروسیسر کنفیگریشن فائل (JSON کے بطور اسٹور کی گئی) کو بھی حب پر موجود ریپو پر اپ لوڈ کیا جائے گا۔

اب فائن ٹیون ہمارے ماڈل کو کال کرکے `train` طریقہ:

```py
>>> train_results = trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

## انفرنس

بہت اچھا، اب جب کہ آپ نے ایک ماڈل ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

انفرنس کے لیے ایک ویڈیو لوڈ کریں:

```py
>>> sample_test_video = next(iter(test_dataset))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif" alt="Teams playing basketball"/>
</div>

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`VideoClassificationPipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ ویڈیو کی درجہ بندی کے لیے ایک `pipeline` فوری بنائیں، اور اپنا ویڈیو اس تک پہنچائیں:

```py
>>> from transformers import pipeline
from accelerate import Accelerator

>>> video_cls = pipeline(model="my_awesome_video_cls_model")
>>> video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
[{'score': 0.9272987842559814, 'label': 'BasketballDunk'},
 {'score': 0.017777055501937866, 'label': 'BabyCrawling'},
 {'score': 0.01663011871278286, 'label': 'BalanceBeam'},
 {'score': 0.009560945443809032, 'label': 'BandMarching'},
 {'score': 0.0068979403004050255, 'label': 'BaseballPitch'}]
```

اگر آپ چاہیں تو آپ `pipeline` کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں۔

```py
>>> def run_inference(model, video):
...     # (num_frames, num_channels, height, width)
...     permuted_sample_test_video = video.permute(1, 0, 2, 3)
...     inputs = {
...         "pixel_values": perumuted_sample_test_video.unsqueeze(0),
...         "labels": torch.tensor(
...             [sample_test_video["label"]]
...         ),  # this can be skipped if you don't have labels available.
...     }

...     device = Accelerator().device
...     inputs = {k: v.to(device) for k, v in inputs.items()}
...     model = model.to(device)

...     # forward pass
...     with torch.no_grad():
...         outputs = model(**inputs)
...         logits = outputs.logits

...     return logits
```

اب، اپنا ان پٹ ماڈل پر دیں اور `logits` واپس کریں:

```py
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

`logits` کو ڈی کوڈ کرتے ہوئے، ہمیں ملتا ہے:

```py
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```
