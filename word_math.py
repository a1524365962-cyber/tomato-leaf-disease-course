"""Minimal native Office Math constructors for the course manuscript."""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def run(text: str, upright: bool = False):
    node=OxmlElement('m:r')
    if upright:
        props=OxmlElement('m:rPr');style=OxmlElement('m:sty');style.set(qn('m:val'),'p')
        props.append(style);node.append(props)
    value=OxmlElement('m:t');value.text=text;node.append(value)
    return node

def sequence(*items):
    node=OxmlElement('m:e')
    for item in items: node.append(run(item) if isinstance(item,str) else item)
    return node

def wrap(tag: str,value):
    node=OxmlElement(tag)
    if isinstance(value,str): node.append(run(value))
    elif value.tag==qn('m:e'):
        for child in list(value): node.append(child)
    else: node.append(value)
    return node

def sub(base,index):
    node=OxmlElement('m:sSub');node.append(wrap('m:e',base));node.append(wrap('m:sub',index));return node

def fraction(top,bottom):
    node=OxmlElement('m:f');node.append(wrap('m:num',top));node.append(wrap('m:den',bottom));return node

def summation(lower,upper,expression):
    node=OxmlElement('m:nary');props=OxmlElement('m:naryPr')
    char=OxmlElement('m:chr');char.set(qn('m:val'),'∑');props.append(char)
    loc=OxmlElement('m:limLoc');loc.set(qn('m:val'),'subSup');props.append(loc)
    node.append(props);node.append(wrap('m:sub',lower));node.append(wrap('m:sup',upper));node.append(wrap('m:e',expression));return node

def append_math(paragraph,*items):
    node=OxmlElement('m:oMath')
    for item in items: node.append(run(item) if isinstance(item,str) else item)
    paragraph._p.append(node);return paragraph
