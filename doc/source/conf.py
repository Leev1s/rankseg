import os
import sys

import shibuya

# for example source
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_code"))

project = "rankseg"
copyright = "Copyright &copy; 2025, Ben Dai"
author = "Ben Dai"

version = shibuya.shibuya_version
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "jupyter_sphinx",
    "sphinx_togglebutton",
    "nbsphinx",
    "numpydoc",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.video",
    "sphinxcontrib.youtube",
    "sphinx_click",
    "sphinx_sqlalchemy",
    "sphinx_contributors",
    "sphinx_prompt",
]
todo_include_todos = True
jupyter_sphinx_thebelab_config = {
    'requestKernel': True,
}
jupyter_sphinx_require_url = ''
nbsphinx_requirejs_path = ''
sitemap_excludes = ['404/']

# extlinks = {
#     'pull': ('https://github.com/lepture/shibuya/pull/%s', 'pull request #%s'),
#     'issue': ('https://github.com/lepture/shibuya/issues/%s', 'issue #%s'),
# }

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ("https://numpy.org/devdocs/", None),
}

templates_path = ["_templates"]
html_static_path = ["_static"]
html_extra_path = ["_public"]

html_title = "rankseg"
html_theme = "shibuya"
html_baseurl = "https://rankseg.github.io/"
sitemap_url_scheme = "{link}"

html_copy_source = False
html_show_sourcelink = False

# html_additional_pages = {
#     "branding": "branding.html",
# }

if os.getenv('USE_DOCSEARCH'):
    extensions.append("sphinx_docsearch")
    docsearch_app_id = "3RU4IG0D1E"
    docsearch_api_key = "ec63fbf7ade2fa535b0b82c86e7d1463"
    docsearch_index_name = "propopt"

if os.getenv("TRIM_HTML_SUFFIX"):
    html_link_suffix = ""

html_favicon = "_static/logo_w.svg"

html_theme_options = {
    "logo_target": "./index.html",
    "light_logo": "_static/logo.svg",
    "dark_logo": "_static/logo_w.svg",

    "og_image_url": "https://rankseg.github.io/icon.png",

    "discussion_url": "https://github.com/lepture/propopt/discussions",
    # "twitter_url": "https://twitter.com/lepture",
    # "github_url": "https://github.com/lepture/propopt",

    # "carbon_ads_code": "CE7DKK3W",
    # "carbon_ads_placement": "propopt",

    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Getting Started",
            "url": "getting_started/index",
            "children": [
                {
                    "title": "Installation",
                    "url": "getting_started/install",
                    "summary": "Installation instructions",
                },
            ]
        },
    ]
}

if "READTHEDOCS" in os.environ:
    html_context = {
        "source_type": "github",
        "source_user": "statmlben",
        "source_repo": "rankseg",
    }
    html_theme_options["carbon_ads_code"] = ""
    html_theme_options["announcement"] = (
        "This documentation is hosted on Read the Docs only for testing. Please use "
        "<a href='https://rankseg.github.io/'>the main documentation</a> instead."
    )
else:
    html_context = {
        "source_type": "github",
        "source_user": "statmlben",
        "source_repo": "rankseg",
        "buysellads_code": "CE7DKK3M",
        "buysellads_placement": "rankseg",
        "buysellads_container_selector": ".yue > section > section",
    }


DEBUG_RTD = False

if DEBUG_RTD:
    os.environ['READTHEDOCS_PROJECT'] = 'rankseg'
    html_context["DEBUG_READTHEDOCS"] = True
    html_theme_options["carbon_ads_code"] = None
