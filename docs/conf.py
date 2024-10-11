extensions = [
    "myst_parser",
    "furo",  # Ensure you have a theme installed
    'sphinx.ext.autodoc',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = "furo"
project = 'pyrodigy'
author = 'Pyro'
version = '0.2.1'
release = version
