# Deep Learning Utils

## Install Dependencies 

-------

>### Google Colab

```python
from utils.external import handlers

handlers.install_dependencies(notebook=True)
```

>### Linux

```python
from utils.external import handlers

handlers.install_dependencies(notebook=False, passwords='???')
```

## Datatset

-------

>###  Kvasir-SEG

```python
from utils.data import kvasir, load

working_dir = 'to_path/'

kvasir.get(working_dir, shape=(224, 224), batch_size=64, dname='kvasir', prefix='data')

data = load.npz_nbatch(working_dir, start=0, end=2, dname='kvasir', prefix='data')

for key in data:

  print(data[key].shape)
```

>### Labeled Faces in the Wild

```python
from utils.data import lfw, load

working_dir = 'to_path/'

lfw.get(working_dir, shape=(224, 224), batch_size=64, dname='lfw', prefix='data')

data = load.npz_nbatch(working_dir, start=0, end=2, dname='lfw', prefix='data')

for key in data:

  print(data[key].shape)
```

>### Cityscapes

```python
import cv2

from utils.external import handlers

handlers.install_dependencies(notebook=True)

from utils.data import cityscapes

username = '???'
password = '???'

working_dir = 'to_path/'

# e.g. {'train': 64, 'test': 64, 'val': 32}
batch_size = {'train': 64}

# [origin] =======================================================
cityscapes.Info.cache_dir = './.cache/cityscapes/origin/left_view'

info = cityscapes.Info.left_view

# i.e. cv_decode = True, runs up to 1.8x faster
cityscapes.get(username, password, info, dest=working_dir, 
               shape=(256, 128), batch_size=batch_size, 
               dname='left_view', prefix='data', 
               cv_decode=True)

# [annotations] =======================================================

cityscapes.Info.cache_dir = './.cache/cityscapes/annotations/panoptic_parts'

alignment_meta = cityscapes.Info.get_alignment_meta(working_dir,
                                                    dname='left_view')

info = cityscapes.Info.panoptic_parts

cityscapes.get(username, password, info, dest=working_dir, 
               shape=(256, 128), batch_size=batch_size, 
               dname='panoptic_parts', prefix='data',
               interpolation=cv2.INTER_NEAREST, 
               alignment_meta=alignment_meta)

```

>### Flowers

```python
from utils.data import flowers, load

working_dir = 'to_path/'

info = flowers.Info.images

# origin
flowers.get(info, dest=working_dir, shape=(224, 224), 
            batch_size=64, dname='flowers',  
            prefix='data', shuffle=True, 
            random_state=None)

# flowers.get(..., dname='flowers', ...)
alignment_meta = flowers.Info.get_alignment_meta(working_dir, 
                                                 dname='flowers')

info = flowers.Info.segments

# segments | i.e. shuffle=True - will be ignored due to alignment
flowers.get(info, dest=working_dir, shape=(224, 224), 
            batch_size=64, dname='flowers_segments',  prefix='data', 
            alignment_meta=alignment_meta)
```