from django import template
import re

register = template.Library()

@register.filter
def get_item(obj, key):
    return obj[key]

@register.filter
def index(sequence, i):
    try:
        return sequence[i]
    except (IndexError, TypeError):
        return 0

@register.filter
def percentage(value, total):
    try:
        return "{:.1f}%".format((value / total) * 100)
    except (ZeroDivisionError, TypeError):
        return "0%"

@register.filter
def remove_leading_number(value):
    """Remove leading number, dot, and space from a string (e.g., '1. Question' -> 'Question')."""
    return re.sub(r'^\d+\.\s*', '', value)
