"""
Author: Harsh Gupta
Created on: 2nd Nov 2020 4:50 PM
"""

import string


def pprint(**kwargs):
    if "title" in kwargs:
        print(kwargs["title"], end="\n")
        del kwargs["title"]
    for k, v in kwargs.items():
        print("==========\n{}\n\n{}".format(k.capitalize(), v), end="\n")
    print("===========================")


def clean_text(text):
    return "".join(x.lower() for x in text if x not in string.punctuation)
