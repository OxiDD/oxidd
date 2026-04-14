"""Configuration file for the Sphinx documentation builder."""

# spell-checker:ignore intersphinx,sphinxcontrib,katex,pydata
# spell-checker:ignore prerender,sourcelink,subclasshook,bysource

import os  # noqa: I001

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import oxidd

project = "OxiDD"
copyright = "2024-2026, OxiDD Contributors"
author = "OxiDD Contributors"
version = oxidd.__version__

gh_ref = os.environ.get("GITHUB_REF", "")
release = version if gh_ref == f"refs/tags/v{version}" else f"{version}.dev"

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "show-inheritance": True,
    "special-members": True,
    "exclude-members": (
        "__class_getitem__, __init_subclass__, __subclasshook__, __weakref__"
    ),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "../../../doc/book/src/img/logo-96x96.jpg"
html_theme_options = {
    "logo": {
        "alt_text": "OxiDD Logo",
        "text": "OxiDD",
        "link": "https://oxidd.net",
    },
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OxiDD/oxidd",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Matrix",
            "url": "https://matrix.to/#/#oxidd:matrix.org",
            "icon": "fa-custom fa-matrix-org",
        },
    ],
    "switcher": {
        "json_url": "https://oxidd.net/api/python/versions.json",
        "version_match": "dev" if gh_ref == "refs/heads/main" else f"v{release}",
    },
    "search_as_you_type": True,
}
html_context = {
    "default_mode": "auto",  # auto dark/light mode
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("matrix-org-icon.js", {"defer": "defer"})]

# Disable links to .rst files
html_show_sourcelink = False
