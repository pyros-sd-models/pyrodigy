extensions = [
    "myst_parser",
    "sphinx_rtd_theme",  # Ensure you have a theme installed
    'sphinx.ext.autodoc',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
project = 'pyrodigy'
author = 'Pyro'
version = '0.1.3'  # Match this with your package version
release = version