from django import template

def get_item(form, key):
    # For Django forms, use form.fields to get the BoundField
    return form[key]

register = template.Library()
register.filter('get_item', get_item)
