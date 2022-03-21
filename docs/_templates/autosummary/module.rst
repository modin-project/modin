{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
   {%- if not item.startswith('_') %}
      {{ item }}
   {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
