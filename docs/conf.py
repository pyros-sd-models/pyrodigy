extensions = [
    "myst_parser",
    "sphinx_rtd_theme",  # Ensure you have a theme installed
    'sphinx.ext.autodoc',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = "furo"
project = 'pyrodigy'
author = 'Pyro'
version = '0.2.0'  # Match this with your package version
release = version
