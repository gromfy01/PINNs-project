#!/usr/bin/env python3
"""Convert overleaf/original.tex (asmejour) into the Springer Nature sn-jnl skeleton
(IJAMT, Engineering with Computers and other Springer journals use the same template).

Usage (from publication/manuscript/springer/):
    python3 convert.py [SRC] [OUT] [--refstyle sn-basic|sn-mathphys-num] [--keep-small]
                       [--keep-clearpage] [--keep-comments] [--flat] [--iicol] [--wide LABEL=MODE ...]

Defaults: SRC = ../overleaf/original.tex, OUT = manuscript_sn.tex, the four wide tables
rotated, three tables with a narrower column gap, the two-column float flush before the
Conclusions removed and all TeX comments stripped (the source carries working notes).

The body (Introduction .. end of Appendix B) is copied verbatim; only the
replacements listed below are applied, each with an asserted count.
"""
import re
import sys

args = sys.argv[1:]
pos = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] not in ('--refstyle', '--wide'))]
src = pos[0] if len(pos) > 0 else '../overleaf/original.tex'
out = pos[1] if len(pos) > 1 else 'manuscript_sn.tex'
refstyle = 'sn-basic,Numbered'
if '--refstyle' in args:
    v = args[args.index('--refstyle') + 1]
    refstyle = 'sn-basic,Numbered' if v == 'sn-basic' else v
keep_small = '--keep-small' in args
no_clearpage = '--keep-clearpage' not in args
strip_comments = '--keep-comments' not in args
flat = '--flat' in args  # submission package: figures and .bib next to the .tex, no subfolders
iicol = '--iicol' in args
keep_star = '--keep-star' in args
# optional table-width fixes, applied by label: --wide LABEL=MODE (MODE: sideways | tabcolsep:<pt> | resize)
wide = {'tab:families': 'tabcolsep:4.9', 'tab:curves': 'tabcolsep:5', 'tab:corrupt': 'tabcolsep:5',
        'tab:main': 'sideways', 'tab:paired': 'sideways', 'tab:backend': 'sideways',
        'tab:admissibility': 'sideways'}
for i, a in enumerate(args):
    if a == '--wide':
        lab, mode = args[i + 1].split('=', 1)
        wide[lab] = mode

lines = open(src, encoding='utf-8').read().split('\n')
text = '\n'.join(lines)


def grab(start_pat, end_pat, inclusive_end=False):
    s = text.index(start_pat)
    e = text.index(end_pat, s)
    if inclusive_end:
        e += len(end_pat)
    return text[s:e]


# ---- pieces taken verbatim from the source ------------------------------
title = re.search(r'^\\title\{(.*)\}\s*$', text, re.M).group(1)
keywords = re.search(r'^\\keywords\{(.*)\}\s*$', text, re.M).group(1)
abstract = grab('\\begin{abstract}', '\\end{abstract}')[len('\\begin{abstract}'):].strip()
body = grab('\\section{Introduction}', '\\begin{nomenclature}')
nomen = grab('\\begin{nomenclature}', '\\end{nomenclature}', inclusive_end=True)
appx = grab('\\appendix', '\n\\selectlanguage{english}')  # the active line, not the one inside a comment

log = []


def sub(s, old, new, expect, what):
    n = s.count(old)
    active = sum(1 for ln in s.split('\n') if old in ln and not ln.lstrip().startswith('%'))
    assert active == expect, (what, old, n, active, expect)
    log.append(f'{what}: {active} active (+{n-active} in comments)  {old!r} -> {new!r}')
    return s.replace(old, new)




def strip_tex_comments(doc):
    """Drop the text of every TeX comment but keep the % itself, so that a comment
    that glued two lines still glues them; lines that held only a comment vanish."""
    out = []
    for ln in doc.split('\n'):
        i, cut = 0, None
        while i < len(ln):
            if ln[i] == '\\':
                i += 2
                continue
            if ln[i] == '%':
                cut = i
                break
            i += 1
        if cut is None:
            out.append(ln)
        elif ln[:cut].strip() == '':
            continue
        else:
            out.append(ln[:cut].rstrip() + '%' if not ln[:cut].endswith((' ', '\t')) else ln[:cut].rstrip())
    return '\n'.join(out)

# ---- body changes --------------------------------------------------------
if not keep_star:
    body = sub(body, '\\begin{table*}', '\\begin{table}', 12, 'starred table -> table (single column)')
    body = sub(body, '\\end{table*}', '\\end{table}', 12, 'starred table -> table (single column)')
    body = sub(body, '\\begin{figure*}', '\\begin{figure}', 6, 'starred figure -> figure (single column)')
    body = sub(body, '\\end{figure*}', '\\end{figure}', 6, 'starred figure -> figure (single column)')

# [!h] / [h!] make LaTeX warn "`!h' float specifier changed to `!ht'", and the two schematics
# then fill a text page on their own ("Text page ... contains only floats"); [htbp] lets them
# go to a float page instead
body = sub(body, '\\begin{table}[!h]', '\\begin{table}[htbp]', 1, 'float specifier [!h] -> [htbp]')
st = '*' if keep_star else ''  # the two schematics are figure* in the source
body = sub(body, '\\begin{figure' + st + '}[h!]', '\\begin{figure' + st + '}[htbp]', 2, 'float specifier [h!] -> [htbp]')

if not keep_small:
    # \small inside the 14 table floats overrides the class table font (8bp); drop it
    n_before = body.count('\\small')
    assert n_before == 14, n_before
    body = re.sub(r'^\\centering\\small\s*$', r'\\centering', body, flags=re.M)
    body = re.sub(r'^\\small\s*$', '', body, flags=re.M)
    assert body.count('\\small') == 0
    log.append('table font: removed 14x \\small inside table floats (class table font 8bp applies)')

if no_clearpage:
    old = re.search(r'^\\clearpage\\clearpage.*$', body, re.M).group(0)
    body = body.replace(old, '% [sn-jnl conversion] \\clearpage\\clearpage removed (two-column float flush)')
    log.append('removed \\clearpage\\clearpage before Conclusions')

# per-table width fixes
for lab, mode in wide.items():
    m = re.search(r'\\begin\{table(\*?)\}(\[[^\]]*\])?((?:(?!\\end\{table\*?\}).)*?\\label\{' + re.escape(lab) + r'\}.*?)\\end\{table\*?\}',
                  body, re.S)
    assert m, lab
    whole = m.group(0)
    star = m.group(1)
    inner = m.group(3)
    if mode == 'sideways':
        # sn-jnl hands an empty placement on to rotating ("No positions in optional float
        # specifier"); a rotated table fills a page of its own anyway
        new = '\\begin{sidewaystable}[p]' + inner + '\\end{sidewaystable}'
    elif mode.startswith('tabcolsep:'):
        pt = mode.split(':')[1]
        # put the setting right after \label{...}
        inner2 = re.sub(r'(\\begin\{tabular)', r'\\setlength{\\tabcolsep}{' + pt + r'pt}%\n\1', inner, count=1)
        new = '\\begin{table' + star + '}' + (m.group(2) or '') + inner2 + '\\end{table' + star + '}'
    elif mode == 'resize':
        inner2 = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', r'\\resizebox{\\textwidth}{!}{%\n\1}', inner, count=1, flags=re.S)
        # sn-jnl's table wraps threeparttable, whose tabular hook breaks inside \resizebox;
        # use the class's saved plain float (tableorg) with the class table font instead
        new = '\\begin{tableorg}' + (m.group(2) or '') + '\\tablebodyfont' + inner2 + '\\end{tableorg}'
    else:
        raise SystemExit('bad mode ' + mode)
    body = body.replace(whole, new)
    log.append(f'table {lab}: {mode}')

# Table 14 (label-corruption results) stands after the Limitations paragraph in the source,
# where asmejour flushes it with \clearpage before the Conclusions. Without that flush it
# drifts into the Conclusions, so it is moved up to just before the Limitations paragraph.
m = re.search(r'\\begin\{table\*?\}(?:(?!\\end\{table\*?\}).)*?\\label\{tab:corrupt\}.*?\\end\{table\*?\}\n', body, re.S)
assert m and body.count('\\paragraph{Limitations}') == 1
assert body.index('\\paragraph{Limitations}') < m.start()
body = body[:m.start()] + body[m.end():]
body = body.replace('\\paragraph{Limitations}', m.group(0) + '\n\\paragraph{Limitations}')
log.append('table tab:corrupt moved before \\paragraph{Limitations}')

# ---- appendices ------------------------------------------------------------
appx = sub(appx, '\\appendix   %%% starting appendices',
           '\\begin{appendices}\n\\counterwithin*{equation}{section}% [sn-jnl conversion] restart (A1), (B1) per appendix as in asmejour',
           1, '\\appendix -> appendices environment + per-appendix equation reset')
appx = appx.rstrip() + '\n\n\\end{appendices}\n'

# ---- assemble ----------------------------------------------------------------
cls_opts = 'pdflatex,' + refstyle + (',iicol' if iicol else '')
preamble = r'''%% Springer Nature LaTeX template (sn-jnl.cls, template Version 3, October 2023),
%% as used by Springer journals such as IJAMT and Engineering with Computers.
%% Generated by springer/convert.py from overleaf/original.tex (asmejour); the body text,
%% equations, tables, figures, nomenclature and appendices are copied verbatim except for
%% the changes listed in the conversion report.
\RequirePackage{fix-cm}% scalable Computer Modern: no font size substitutions at the bp sizes of sn-jnl
\documentclass[''' + cls_opts + r''']{sn-jnl}

%% packages of the Springer template skeleton that the paper needs
\usepackage{graphicx}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{booktabs}%

%% PDF bookmarks down to subsubsections. \paragraph and \bmhead headings follow a
%% (sub)section directly and would make hyperref warn about skipped bookmark levels.
\hypersetup{bookmarksdepth=3}

%% from the preamble of original.tex
\usepackage{tabularx}
\usepackage{accents}
\newcommand{\edot}{\accentset{\scalebox{0.4}{$\bullet$}}{\varepsilon}}
\newcommand{\pending}[1]{{\color{red}[#1]}}

%% asmejour provides the nomenclature environment with \entry and \EntryHeading;
%% sn-jnl has no counterpart, so an equivalent list is defined here
%% (symbol right-aligned and followed by " = ", hanging description, bold subheadings).
\newlength{\nomenlabelwidth}
\makeatletter
\newenvironment{nomenclature}[1][Nomenclature]{%
  \section*{#1}%
  \settowidth{\nomenlabelwidth}{$\sigma_{rr},\sigma_{\theta\theta},\sigma_{zz}$${}={}$}%
  \begin{list}{}{%
    \setlength{\topsep}{0pt}\setlength{\partopsep}{0pt}%
    \setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}%
    \setlength{\labelsep}{0pt}\setlength{\labelwidth}{\nomenlabelwidth}%
    \setlength{\leftmargin}{\nomenlabelwidth}%
    \let\makelabel\@mklab}\raggedright}%
  {\end{list}}
\newcommand{\entry}[2]{\item[\hfill#1${}={}$]#2}
\newcommand{\EntryHeading}[1]{\itemsep8pt\goodbreak\item[\bfseries#1\hfill]\itemsep0pt\@itempenalty=1000\relax}
\makeatother

\raggedbottom
\graphicspath{{../overleaf/}}% figures stay in overleaf/pictures (single source)

\begin{document}

\title{''' + title + r'''}

\author*[1]{\fnm{Dmitriy} \sur{Demin}}\email{ddemin@hse.ru}
\author[1]{\fnm{Georgii} \sur{Romanenko}}\email{Georgii.v.romanenko@gmail.com}

\affil*[1]{\orgname{HSE University}, \orgaddress{\street{11 Pokrovsky Boulevard}, \city{Moscow}, \country{Russia}}}

\abstract{\unboldmath ''' + abstract + r'''}

\keywords{''' + keywords + r'''}

\maketitle

'''

declarations = r'''
\backmatter

\section*{Declarations}

\bmhead{Conflict of interest} The authors declare no competing interests.

\bmhead{Data availability} The finite-element data of this study, the 2443 raw Abaqus reports and the 112 history reports used to verify the drawing stress, are deposited in Zenodo at \url{https://doi.org/10.5281/zenodo.23352871}.

\bmhead{Code availability} The code, the log of all training runs and the verification scripts are available in Zenodo at \url{https://doi.org/10.5281/zenodo.23352452} and on GitHub at \url{https://github.com/gromfy01/wire-drawing-pinn-surrogates}.

'''

doc = (preamble + body.rstrip() + '\n\n' + nomen + '\n' + declarations + appx +
       '\n\\bibliography{../overleaf/asmejour-sample}\n\n\\end{document}\n')
if flat:
    n = doc.count('{pictures/')
    doc = doc.replace('{pictures/', '{')
    doc = doc.replace('\\graphicspath{{../overleaf/}}% figures stay in overleaf/pictures (single source)\n', '')
    doc = doc.replace('\\bibliography{../overleaf/asmejour-sample}', '\\bibliography{references}')
    log.append(f'flat package: {n} figure paths without pictures/, bibliography references.bib')
if strip_comments:
    doc = strip_tex_comments(doc)
    log.append('TeX comments stripped')
open(out, 'w', encoding='utf-8').write(doc)
print('\n'.join(log))
print('wrote', out, 'class options:', cls_opts)
