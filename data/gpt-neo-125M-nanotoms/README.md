# GPT-Neo 125M Nanotoms

## Model description

This transformer model is based on the
[GPT-Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125M) model and has been
trained for text generation.

## Training data

The model was trained using object descriptions data from the
[Narrative Atoms](https://github.com/kingsdigitallab/nanotoms) project.

## How to use

Clone/download a copy of this repository.

```python
>>> from happytransformer import HappyGeneration
>>> gen = HappyGeneration(load_path=".")
>>> gen.generate_text("The middling society ")

GenerationResult(text=' \nof London, 1650-1680.  \nThis is the manuscript  \nof Thomas Elyon,  \na man of about 15 years of age.  \nThis manuscript is a copy of  \n')
```

This example uses the [happytransformer](https://happytransformer.com/) package.
