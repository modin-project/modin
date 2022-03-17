{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}

.. autosummary::
    :members:
    :toctree:
    {% for item in all_methods %}
    {%- if not item.startswith('_') or item in ['__call__'] %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}

.. autosummary::
    :members:
    :toctree:
    {% for item in all_attributes %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}
