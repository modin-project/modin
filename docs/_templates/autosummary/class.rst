{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
    :toctree:
    {% for item in all_methods %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
    :toctree:
    {% for item in all_attributes %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}
{% endblock %}
