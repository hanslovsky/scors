import os

project = "scors"
copyright = "2026, Philipp Hanslovsky"  # noqa: A001
author = "Philipp Hanslovsky"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

exclude_patterns = ["_build", ".DS_Store", ".venv", "public"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "default_mode": "auto",
}
