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

# گہرائی کا تخمینہ

مونوکولر گہرائی کا تخمینہ ایک کمپیوٹر وژن کا کام ہے جس میں ایک تصویر سے کسی منظر کی گہرائی کی معلومات کی پیش گوئی کرنا شامل ہے۔ دوسرے لفظوں میں، یہ ایک ہی کیمرے کے نقطہ نظر سے کسی منظر میں اشیاء کے فاصلے کا اندازہ لگانے کا عمل ہے۔

مونوکولر گہرائی کے تخمینے میں مختلف ایپلی کیشنز ہیں، بشمول 3D تعمیر نو، بڑھا ہوا حقیقت، خود مختار ڈرائیونگ، اور روبوٹکس۔ یہ ایک چیلنجنگ کام ہے کیونکہ اس میں منظر میں موجود اشیاء اور متعلقہ گہرائی سے متعلق معلومات کے درمیان پیچیدہ تعلقات کو سمجھنے کے لیے ماڈل کی ضرورت ہوتی ہے، جو روشنی کے حالات، رکاوٹ اور ساخت جیسے عوامل سے متاثر ہو سکتے ہیں۔

گہرائی کے تخمینے کے دو اہم زمرے ہیں:

- **مطلق گہرائی کا تخمینہ**: اس ٹاسک ویرینٹ کا مقصد کیمرے سے گہرائی کی درست پیمائش فراہم کرنا ہے۔ یہ اصطلاح میٹرک گہرائی کے تخمینے کے ساتھ ایک دوسرے کے ساتھ استعمال ہوتی ہے، جہاں گہرائی کو میٹر یا فٹ میں درست پیمائش میں فراہم کیا جاتا ہے۔ گہرائی کا مطلق تخمینہ ماڈلز آؤٹ پٹ گہرائی کے نقشے عددی اقدار کے ساتھ جو حقیقی دنیا کے فاصلے کی نمائندگی کرتے ہیں۔

- **متعلقہ گہرائی کا تخمینہ**: رشتہ دار گہرائی کے تخمینے کا مقصد عین پیمائش فراہم کیے بغیر کسی منظر میں اشیاء یا پوائنٹس کی گہرائی کی ترتیب کا اندازہ لگانا ہے۔ یہ ماڈلز ایک گہرائی کا نقشہ نکالتے ہیں جو اس بات کی نشاندہی کرتا ہے کہ منظر کے کون سے حصے ایک دوسرے کے قریب یا زیادہ رشتہ دار ہیں بغیر A اور B کے اصل فاصلے کے۔

اس گائیڈ میں، ہم دیکھیں گے کہ [Depth Anything V2 سے کیسے اندازہ لگایا جائے۔](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)، ایک جدید ترین زیرو شاٹ رشتہ دار گہرائی کا تخمینہماڈل، اور [ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth)، ایک مکمل گہرائی کا تخمینہماڈل.

<Tip>

تمام ہم آہنگ فن تعمیرات اور چیک پوائنٹس دیکھنے کے لیے [گہرائی کا تخمینہ](https://huggingface.co/tasks/depth-estimation) ٹاسک صفحہ چیک کریں۔

</Tip>

شروع کرنے سے پہلے، ہمیں Transformers کا تازہ ترین ورژن انسٹال کرنا ہوگا:

```bash
pip install -q -U transformers
```

## گہرائی کا تخمینہ لگانے والی پائپ لائن

ماڈل معاون گہرائی کے تخمینے کے ساتھ انفرنس کو آزمانے کا آسان ترین طریقہ متعلقہ [`pipeline`] استعمال کرنا ہے۔ Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads) پر [چیک پوائنٹ سے ایک پائپ لائن شروع کریں:

```py
>>> from transformers import pipeline
from accelerate import Accelerator
>>> import torch
>>> device = Accelerator().device
>>> checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
>>> pipe = pipeline("depth-estimation", model=checkpoint, device=device)
```

اگلا، تجزیہ کرنے کے لیے ایک تصویر منتخب کریں:

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="Photo of a bee"/>
</div>

تصویر کو پائپ لائن پر منتقل کریں۔

```py
>>> predictions = pipe(image)
```

پائپ لائن دو اندراجات کے ساتھ ایک لغت لوٹاتی ہے۔ پہلا، جسے `predicted_depth` کہا جاتا ہے، ایک ٹینسر ہے جس کی قدریں ہر پکسل کے میٹر میں ظاہر کی گئی گہرائی ہیں۔ دوسرا، `depth`، ایک PIL امیج ہے جو گہرائی کے تخمینے کے نتائج کا تصور کرتی ہے۔

آئیے تصوراتی نتائج پر ایک نظر ڈالتے ہیں:

```py
>>> predictions["depth"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="Depth estimation visualization"/>
</div>

## گہرائی کا تخمینہ انفرنس ہاتھ سے

اب جب کہ آپ نے گہرائی کا تخمینہ لگانے والی پائپ لائن کو استعمال کرنے کا طریقہ دیکھا ہے، آئیے دیکھتے ہیں کہ ہم اسی نتیجے کو ہاتھ سے کیسے نقل کر سکتے ہیں۔

ماڈل اور منسلک پروسیسر کو Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads) پر [چیک پوائنٹ سے لوڈ کر کے شروع کریں۔ یہاں ہم پہلے کی طرح ہی چیک پوائنٹ استعمال کریں گے:

```py
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "Intel/zoedepth-nyu-kitti"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
```

`image_processor` کا استعمال کرتے ہوئے ماڈل کے لیے امیج ان پٹ تیار کریں جو تصویر کی ضروری تبدیلیوں جیسے کہ سائز تبدیل کرنا اور نارملائزیشن کا خیال رکھے گا:

```py
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
```

تیار کردہ ان پٹ کو ماڈل کے ذریعے منتقل کریں:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(pixel_values)
```

آئیے کسی بھی پیڈنگ کو ہٹانے اور اصل تصویر کے سائز سے ملنے کے لیے گہرائی کے نقشے کا سائز تبدیل کرنے کے لیے نتائج کو پوسٹ پروسیس کریں۔ `post_process_depth_estimation` `"predicted_depth"` پر مشتمل ڈکٹوں کی فہرست تیار کرتا ہے۔

```py
>>> # ZoeDepth dynamically pads the input image. Thus we pass the original image size as argument
>>> # to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
>>> depth = depth.detach().cpu().numpy() * 255
>>> depth = Image.fromarray(depth.astype("uint8"))
```

<Tip>
<p>میں<a href="https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131">اصل نفاذ</a>زو گہرائیماڈلانجام دیتا ہےانفرنساصل اور پلٹی ہوئی تصاویر دونوں پر اور نتائج کا اوسط۔ دی<code>post_process_depth_estimation</code>فنکشن ہمارے لیے فلپ شدہ آؤٹ پٹ کو اختیاری میں منتقل کر کے اسے سنبھال سکتا ہے۔<code>outputs_flipped</code>دلیل:</p> <pre><code class="language-Python">>>> torch.no_grad(): ... آؤٹ پٹ =ماڈل(pixel_values) ... outputs_flipped =ماڈل(pixel_values=torch.flip(inputs.pixel_values, dims=[3])) >>> post_processed_output = image_processor.post_process_depth_estimation( ... outputs, ... source_sizes=[(image.height, image.width)],_fpped output ...=lipped output ...
</code></pre>
</Tip>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization-zoe.png" alt="Depth estimation visualization"/>
</div>
