# Deep Learning Utils

## Datatset

-------

>###  Kvasir-SEG

```python
from .data import kvasir, load

kvasir.get(working_dir, shape=(224, 224), batch_size=64, dname='kvasir', prefix='data')

data = load.npz_nbatch(working_dir, start=0, end=2, dname='kvasir', prefix='data')

for key in data:

  print(data[key].shape)
```

>### Labeled Faces in the Wild

```python
from .data import lfw, load

lfw.get(working_dir, shape=(224, 224), batch_size=64, dname='lfw', prefix='data')

data = load.npz_nbatch(working_dir, start=0, end=2, dname='lfw', prefix='data')

for key in data:

  print(data[key].shape)
```
