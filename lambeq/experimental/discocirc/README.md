# DisCoCirc extension for lambeq

Functionality to convert text into DisCoCirc string diagrams, using lambeq's grammar backend.

The DisCoCirc framework was first introduced in the paper [\[Coe10\]](https://arxiv.org/abs/1904.03478). For further theoretical and experimental work on DisCoCirc respectively, see [\[LMC24\]](https://arxiv.org/abs/2408.06061) and [\[DBM+24\]](https://arxiv.org/abs/2409.08777).

## Installation

Installing the experimental subpackage requires Python 3.10.

```bash
git clone git@github.com:quantinuum/lambeq.git
cd lambeq
pip install ".[experimental]"
```

## Usage

To get DisCoCirc diagrams using frames:

```python
from lambeq.experimental.discocirc import DisCoCircReader

reader = DisCoCircReader()
reader.text2circuit('Alice likes Bob. Bob likes Alice too.').draw()
```

To get DisCoCirc diagrams with frames decomposed into multiple boxes:

```python
reader.text2circuit('Alice likes Bob. Bob likes Alice too.', sandwich=True).draw()
```
