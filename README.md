# λambeq

![Build status](https://github.com/quantinuum/lambeq/actions/workflows/build_test.yml/badge.svg)
[![License](https://img.shields.io/github/license/quantinuum/lambeq)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/lambeq)](//pypi.org/project/lambeq)
[![PyPI downloads](https://img.shields.io/pypi/dm/lambeq)](//pypi.org/project/lambeq)
[![arXiv](https://img.shields.io/badge/arXiv-2110.04236-green)](//arxiv.org/abs/2110.04236)

## About

lambeq is a toolkit for quantum natural language processing (QNLP).

- Documentation: https://docs.quantinuum.com/lambeq/.
- User support: <lambeq-support@quantinuum.com>.
- Contributions: Please read [our guide](https://docs.quantinuum.com/lambeq/CONTRIBUTING.html).
- If you want to subscribe to lambeq's mailing list, let us know by sending an email to <lambeq-support@quantinuum.com>.

## Getting started

### Prerequisites

- Python 3.10+

### Installation

lambeq can be installed with the command:

```bash
pip install lambeq
```

The default installation of lambeq includes Bobcat parser, a state-of-the-art statistical parser (see [related paper](https://arxiv.org/abs/2109.10044)) fully integrated with the toolkit.

To install lambeq with optional dependencies for extra features, run:

```bash
pip install lambeq[extras]
```

To install lambeq with optional dependencies for experimental features, run:

```bash
pip install lambeq[experimental]
```

To enable DepCCG support, you will need to install the external parser separately.

---
**Note:** The DepCCG-related functionality is no longer actively supported in `lambeq`, and may not work as expected. We strongly recommend using the default Bobcat parser which comes as part of `lambeq`.

---

If you still want to use DepCCG, for example because you plan to apply ``lambeq`` on Japanese, you can install DepCCG separately following the instructions on the [DepCCG homepage](//github.com/masashi-y/depccg). After installing DepCCG, you can download its model by using the script provided in the `contrib` folder of this repository:

```bash
python contrib/download_depccg_model.py
```

## Usage

The [docs/examples](//github.com/quantinuum/lambeq-docs/tree/main/docs/examples)
directory in lambeq's [documentation repository](https://github.com/quantinuum/lambeq-docs) contains notebooks demonstrating usage of the various tools in lambeq.

Example - parsing a sentence into a diagram (see
[docs/examples/parser.ipynb](//github.com/quantinuum/lambeq-docs/blob/main/docs/examples/parser.ipynb)):

```python
from lambeq import BobcatParser

parser = BobcatParser()
diagram = parser.sentence2diagram('This is a test sentence')
diagram.draw()
```

## Testing

Run all tests with the command:

```bash
pytest
```

Note: if you have installed lambeq in a virtual environment, remember to
install pytest in the same environment using pip.

## License

Distributed under the Apache 2.0 license. See [`LICENSE`](LICENSE) for
more details.

## Citation

If you wish to attribute our work, please cite
[the accompanying paper](//arxiv.org/abs/2110.04236):

```
@article{kartsaklis2021lambeq,
   title={lambeq: {A}n {E}fficient {H}igh-{L}evel {P}ython {L}ibrary for {Q}uantum {NLP}},
   author={Dimitri Kartsaklis and Ian Fan and Richie Yeung and Anna Pearson and Robin Lorenz and Alexis Toumi and Giovanni de Felice and Konstantinos Meichanetzidis and Stephen Clark and Bob Coecke},
   year={2021},
   journal={arXiv preprint arXiv:2110.04236},
}
```
