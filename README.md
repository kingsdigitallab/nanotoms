# Narrative Atoms

The Narrative Atoms project contains a set of tools for topic extraction and text generation,
using as a test case data about objects from the [Middling Culture](https://middlingculture.com/) project.

This repository provides a command line tool and a web interface to process and interact with the data.

For the list of changes to the project see the [Changelog](CHANGELOG.md).

## Workflow

[![](https://mermaid.ink/img/eyJjb2RlIjoiZmxvd2NoYXJ0IExSXG4gICAgZGF0YV9yYXdbL1JhdyBkYXRhL10gLS0-IHByZXBhcmUoQ2xlYW4vcHJlcGFyZSlcbiAgICBkYXRhX3JhdyAtLi0gY29tbWVudF9kYXRhX3Jhd1tDU1YgZGF0YSBjdXJhdGVkIGFuZFxcbiBwcm92aWRlZCBieSB0aGUgcmVzZWFyY2ggdGVhbVxcbiB3aXRoIGluZm9ybWF0aW9uIGFib3V0IG9iamVjdHNcXG4gZnJvbSBYVkktWFZJSSBwZXJpb2RdXG4gICAgY2xhc3MgY29tbWVudF9kYXRhX3JhdyBjb21tZW50XG5cbiAgICBwcmVwYXJlIC0tPiBkYXRhX3ByZXBhcmVkWy9QcmVwYXJlZCBkYXRhL11cbiAgICBwcmVwYXJlIC0tPiBkYXRhX3NjcmFwZWRbL1NjcmFwZWQvZXh0cmFjdGVkIGRhdGEvXVxuICAgIHByZXBhcmUgLS4tIGNvbW1lbnRfcHJlcGFyZVtDbGVhbiB0aGUgZGF0YSxcXG4gdmFsaWRhdGUvcGFyc2UgVVJMcyxcXG4gc2NyYXBlIFVSTHNdXG4gICAgY2xhc3MgY29tbWVudF9wcmVwYXJlIGNvbW1lbnRcblxuICAgIGRhdGFfcHJlcGFyZWQgLS4tIGNvbW1lbnRfZGF0YV9wcmVwYXJlZFtDU1Ygd2l0aCBjbGVhbmVkIGRhdGFdXG4gICAgZGF0YV9wcmVwYXJlZCAtLT4gdHJhbnNmb3JtKFRyYW5zZm9ybSlcbiAgICBjbGFzcyBjb21tZW50X2RhdGFfcHJlcGFyZWQgY29tbWVudFxuXG4gICAgZGF0YV9zY3JhcGVkIC0tPiB0cmFuc2Zvcm0oVHJhbnNmb3JtKVxuICAgIGRhdGFfc2NyYXBlZCAtLi0gY29tbWVudF9kYXRhX3NjcmFwcGVkW0NTViB3aXRoIGRhdGEgc2NyYXBlZCBmcm9tIHRoZSBVUkxzXVxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV9zY3JhcHBlZCBjb21tZW50XG5cbiAgICB0cmFuc2Zvcm0gLS4tIGNvbW1lbnRfdHJhbnNmb3JtW01lcmdlIHRoZSBzY3JhcGVkIGFuZCBwcmVwYXJlZCBkYXRhLCBcXG5hZGQgZmVhdHVyZXMgdG8gdGhlIHByZXBhcmVkIGRhdGEsXFxuIGFwcGx5IG5hbWVkIGVudGl0eSByZWNvZ25pdGlvbl1cbiAgICB0cmFuc2Zvcm0gLS0-IGRhdGFfdHJhbnNmb3JtZWRbL1RyYW5zZm9ybWVkIGRhdGEvXVxuICAgIHRyYW5zZm9ybSAtLT4gZGF0YV90cmFpbl9ncHRbL0dQVCB0cmFpbmluZyBkYXRhL10gIFxuICAgIGNsYXNzIGNvbW1lbnRfdHJhbnNmb3JtIGNvbW1lbnRcblxuICAgIGRhdGFfdHJhbnNmb3JtZWQgLS4tIGNvbW1lbnRfZGF0YV90cmFuc2Zvcm1lZFtDU1Ygd2l0aCB0cmFuc2Zvcm1lZCBkYXRhLCBpbmNsdWRpbmdcXG4gdGhlIHNjcmFwcGVkIGRhdGEgYW5kIGV4dHJhY3RlZCBlbnRpdGllc11cbiAgICBkYXRhX3RyYW5zZm9ybWVkIC0tPiB0cmFpbihUcmFpbi90dW5lIHRvcGljIG1vZGVsKVxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV90cmFuc2Zvcm1lZCBjb21tZW50XG5cbiAgICB0cmFpbiAtLT4gZGF0YV9maW5hbFsvVHJhaW5lZCBkYXRhL11cbiAgICB0cmFpbiAtLT4gdG9waWNfbW9kZWxbL1RyYWluZWQgdG9waWMgbW9kZWwvXVxuICAgIHRyYWluIC0uLSBjb21tZW50X3RyYWluW1RyYWluIHRvcGljIG1vZGVsIGFuZCBleHRyYWN0IHRvcGljcyBmcm9tIHRoZSBkYXRhXVxuICAgIGRhdGFfZmluYWwgLS4tIGNvbW1lbnRfZGF0YV9maW5hbFtDU1Ygd2l0aCBmaW5hbCBkYXRhLCBpbmNsdWRpbmcgdG9waWNzXVxuICAgIHRvcGljX21vZGVsIC0uLSBjb21tZW50X3RvcGljX21vZGVsW1RvcGljIG1vZGVsIGZvciB0b3BpYyBleHRyYWN0aW9uXVxuICAgIGNsYXNzIGNvbW1lbnRfdHJhaW4gY29tbWVudFxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV9maW5hbCBjb21tZW50XG4gICAgY2xhc3MgY29tbWVudF90b3BpY19tb2RlbCBjb21tZW50XG5cbiAgICBkYXRhX3RyYWluX2dwdCAtLT4gZ3B0KFRyYWluIEdQVCBtb2RlbClcbiAgICBkYXRhX3RyYWluX2dwdCAtLi0gY29tbWVudF9kYXRhX3RyYWluX2dwdFtUZXh0IGZpbGUgY29udGFpbmluZyBhbGwgdGhlIG9iamVjdHMgZGVzY3JpcHRpb25zXVxuICAgIGdwdCAtLi0gY29tbWVudF9ncHRbTm90ZWJvb2sgdG8gdHJhaW4gR1BUIG1vZGVsIGZvciB0ZXh0IGdlbmVyYXRpb25dXG4gICAgY2xhc3MgY29tbWVudF9kYXRhX3RyYWluX2dwdCBjb21tZW50XG4gICAgY2xhc3MgY29tbWVudF9ncHQgY29tbWVudFxuICAgIFxuICAgIGdwdCAtLT4gZ3B0X21vZGVsWy9UcmFpbmVkIEdQVCBtb2RlbC9dXG4gICAgZ3B0X21vZGVsIC0uLSBjb21tZW50X2dwdF9tb2RlbFtHUFQgbW9kZWwgZm9yIHRleHQgZ2VuZXJhdGlvbl1cbiAgICBjbGFzcyBjb21tZW50X2dwdF9tb2RlbCBjb21tZW50XG5cbiAgICBjbGljayBncHQgXCJodHRwczovL2NvbGFiLnJlc2VhcmNoLmdvb2dsZS5jb20vZHJpdmUvMUNIQnlGR2MyTEtTUGFXNlg2X01xWHFKUm1Xbmxlc2pBXCIgXCJUcmFpbiBHUFQgbW9kZWxcIlxuICAgIGNsaWNrIGdwdF9tb2RlbCBcImh0dHBzOi8vZ2l0aHViLmNvbS9raW5nc2RpZ2l0YWxsYWIvZ3B0LW5lby0xMjVNLW5hbm90b21zXCIgXCJHUFQgbW9kZWxcIlxuXG4gICAgY2xhc3NEZWYgY29tbWVudCBmaWxsOmxpZ2h0eWVsbG93LHN0cm9rZS13aWR0aDowcHg7IiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/edit/#eyJjb2RlIjoiZmxvd2NoYXJ0IExSXG4gICAgZGF0YV9yYXdbL1JhdyBkYXRhL10gLS0-IHByZXBhcmUoQ2xlYW4vcHJlcGFyZSlcbiAgICBkYXRhX3JhdyAtLi0gY29tbWVudF9kYXRhX3Jhd1tDU1YgZGF0YSBjdXJhdGVkIGFuZFxcbiBwcm92aWRlZCBieSB0aGUgcmVzZWFyY2ggdGVhbVxcbiB3aXRoIGluZm9ybWF0aW9uIGFib3V0IG9iamVjdHNcXG4gZnJvbSBYVkktWFZJSSBwZXJpb2RdXG4gICAgY2xhc3MgY29tbWVudF9kYXRhX3JhdyBjb21tZW50XG5cbiAgICBwcmVwYXJlIC0tPiBkYXRhX3ByZXBhcmVkWy9QcmVwYXJlZCBkYXRhL11cbiAgICBwcmVwYXJlIC0tPiBkYXRhX3NjcmFwZWRbL1NjcmFwZWQvZXh0cmFjdGVkIGRhdGEvXVxuICAgIHByZXBhcmUgLS4tIGNvbW1lbnRfcHJlcGFyZVtDbGVhbiB0aGUgZGF0YSxcXG4gdmFsaWRhdGUvcGFyc2UgVVJMcyxcXG4gc2NyYXBlIFVSTHNdXG4gICAgY2xhc3MgY29tbWVudF9wcmVwYXJlIGNvbW1lbnRcblxuICAgIGRhdGFfcHJlcGFyZWQgLS4tIGNvbW1lbnRfZGF0YV9wcmVwYXJlZFtDU1Ygd2l0aCBjbGVhbmVkIGRhdGFdXG4gICAgZGF0YV9wcmVwYXJlZCAtLT4gdHJhbnNmb3JtKFRyYW5zZm9ybSlcbiAgICBjbGFzcyBjb21tZW50X2RhdGFfcHJlcGFyZWQgY29tbWVudFxuXG4gICAgZGF0YV9zY3JhcGVkIC0tPiB0cmFuc2Zvcm0oVHJhbnNmb3JtKVxuICAgIGRhdGFfc2NyYXBlZCAtLi0gY29tbWVudF9kYXRhX3NjcmFwcGVkW0NTViB3aXRoIGRhdGEgc2NyYXBlZCBmcm9tIHRoZSBVUkxzXVxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV9zY3JhcHBlZCBjb21tZW50XG5cbiAgICB0cmFuc2Zvcm0gLS4tIGNvbW1lbnRfdHJhbnNmb3JtW01lcmdlIHRoZSBzY3JhcGVkIGFuZCBwcmVwYXJlZCBkYXRhLCBcXG5hZGQgZmVhdHVyZXMgdG8gdGhlIHByZXBhcmVkIGRhdGEsXFxuIGFwcGx5IG5hbWVkIGVudGl0eSByZWNvZ25pdGlvbl1cbiAgICB0cmFuc2Zvcm0gLS0-IGRhdGFfdHJhbnNmb3JtZWRbL1RyYW5zZm9ybWVkIGRhdGEvXVxuICAgIHRyYW5zZm9ybSAtLT4gZGF0YV90cmFpbl9ncHRbL0dQVCB0cmFpbmluZyBkYXRhL10gIFxuICAgIGNsYXNzIGNvbW1lbnRfdHJhbnNmb3JtIGNvbW1lbnRcblxuICAgIGRhdGFfdHJhbnNmb3JtZWQgLS4tIGNvbW1lbnRfZGF0YV90cmFuc2Zvcm1lZFtDU1Ygd2l0aCB0cmFuc2Zvcm1lZCBkYXRhLCBpbmNsdWRpbmdcXG4gdGhlIHNjcmFwcGVkIGRhdGEgYW5kIGV4dHJhY3RlZCBlbnRpdGllc11cbiAgICBkYXRhX3RyYW5zZm9ybWVkIC0tPiB0cmFpbihUcmFpbi90dW5lIHRvcGljIG1vZGVsKVxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV90cmFuc2Zvcm1lZCBjb21tZW50XG5cbiAgICB0cmFpbiAtLT4gZGF0YV9maW5hbFsvVHJhaW5lZCBkYXRhL11cbiAgICB0cmFpbiAtLT4gdG9waWNfbW9kZWxbL1RyYWluZWQgdG9waWMgbW9kZWwvXVxuICAgIHRyYWluIC0uLSBjb21tZW50X3RyYWluW1RyYWluIHRvcGljIG1vZGVsIGFuZCBleHRyYWN0IHRvcGljcyBmcm9tIHRoZSBkYXRhXVxuICAgIGRhdGFfZmluYWwgLS4tIGNvbW1lbnRfZGF0YV9maW5hbFtDU1Ygd2l0aCBmaW5hbCBkYXRhLCBpbmNsdWRpbmcgdG9waWNzXVxuICAgIHRvcGljX21vZGVsIC0uLSBjb21tZW50X3RvcGljX21vZGVsW1RvcGljIG1vZGVsIGZvciB0b3BpYyBleHRyYWN0aW9uXVxuICAgIGNsYXNzIGNvbW1lbnRfdHJhaW4gY29tbWVudFxuICAgIGNsYXNzIGNvbW1lbnRfZGF0YV9maW5hbCBjb21tZW50XG4gICAgY2xhc3MgY29tbWVudF90b3BpY19tb2RlbCBjb21tZW50XG5cbiAgICBkYXRhX3RyYWluX2dwdCAtLT4gZ3B0KFRyYWluIEdQVCBtb2RlbClcbiAgICBkYXRhX3RyYWluX2dwdCAtLi0gY29tbWVudF9kYXRhX3RyYWluX2dwdFtUZXh0IGZpbGUgY29udGFpbmluZyBhbGwgdGhlIG9iamVjdHMgZGVzY3JpcHRpb25zXVxuICAgIGdwdCAtLi0gY29tbWVudF9ncHRbTm90ZWJvb2sgdG8gdHJhaW4gR1BUIG1vZGVsIGZvciB0ZXh0IGdlbmVyYXRpb25dXG4gICAgY2xhc3MgY29tbWVudF9kYXRhX3RyYWluX2dwdCBjb21tZW50XG4gICAgY2xhc3MgY29tbWVudF9ncHQgY29tbWVudFxuICAgIFxuICAgIGdwdCAtLT4gZ3B0X21vZGVsWy9UcmFpbmVkIEdQVCBtb2RlbC9dXG4gICAgZ3B0X21vZGVsIC0uLSBjb21tZW50X2dwdF9tb2RlbFtHUFQgbW9kZWwgZm9yIHRleHQgZ2VuZXJhdGlvbl1cbiAgICBjbGFzcyBjb21tZW50X2dwdF9tb2RlbCBjb21tZW50XG5cbiAgICBjbGljayBncHQgXCJodHRwczovL2NvbGFiLnJlc2VhcmNoLmdvb2dsZS5jb20vZHJpdmUvMUNIQnlGR2MyTEtTUGFXNlg2X01xWHFKUm1Xbmxlc2pBXCIgXCJUcmFpbiBHUFQgbW9kZWxcIlxuICAgIGNsaWNrIGdwdF9tb2RlbCBcImh0dHBzOi8vZ2l0aHViLmNvbS9raW5nc2RpZ2l0YWxsYWIvZ3B0LW5lby0xMjVNLW5hbm90b21zXCIgXCJHUFQgbW9kZWxcIlxuXG4gICAgY2xhc3NEZWYgY29tbWVudCBmaWxsOmxpZ2h0eWVsbG93LHN0cm9rZS13aWR0aDowcHg7IiwibWVybWFpZCI6IntcbiAgXCJ0aGVtZVwiOiBcImRlZmF1bHRcIlxufSIsInVwZGF0ZUVkaXRvciI6ZmFsc2UsImF1dG9TeW5jIjp0cnVlLCJ1cGRhdGVEaWFncmFtIjpmYWxzZX0)

## Set up

Install [poetry](https://python-poetry.org/docs/#installation) and the requirements:

    poetry install

## Run the cli

    poetry run python cli.py

## Run the gui

    poetry run streamlit run streamlit_app.py

## Development

    poetry install --dev
    poetry shell
