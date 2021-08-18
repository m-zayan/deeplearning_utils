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

from utils.data import cityscapes, load

username = '???'
password = '???'

working_dir = 'to_path/'

# e.g. cityscapes.Info.left_view
info = cityscapes.Info.panoptic_parts

cityscapes.get(username, password, info, dest=working_dir, 
               shape=(256, 128), batch_size=None, dname='cityscapes', 
               prefix='data', interpolation=cv2.INTER_NEAREST)


data = load.npz_nbatch(working_dir, start=0, end=2, 
                       dname='cityscapes/train', prefix='data')

for key in data:

  print(data[key].shape)
```

