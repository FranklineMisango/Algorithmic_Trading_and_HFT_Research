%% arithmaxresearch.cls
%% Arithmax Research Unified LaTeX Template
%% Copyright (c) 2025 Arithmax Research
%% 
%% This class provides a unified template for all Arithmax Research papers
%% with consistent branding, formatting, and package management.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{arithmaxresearch}[2025/10/05 v1.0 Arithmax Research Paper Template]

% Base class options
\newif\if@twocolumn\@twocolumnfalse
\newif\if@onecolumn\@onecolumnfalse
\newif\if@ieee\@ieeefalse
\newif\if@article\@articlefalse

% Class options
\DeclareOption{twocolumn}{\@twocolumntrue\@ieeetrue}
\DeclareOption{onecolumn}{\@onecolumntrue}
\DeclareOption{ieee}{\@ieeetrue}
\DeclareOption{article}{\@articletrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

% Load base class based on options
\if@ieee
    \if@twocolumn
        \LoadClass[twocolumn,11pt]{IEEEtran}
        \IEEEoverridecommandlockouts
    \else
        \LoadClass[onecolumn,11pt]{IEEEtran}
        \IEEEoverridecommandlockouts
    \fi
\else
    \LoadClass[12pt]{article}
\fi

% Required packages
\RequirePackage{amsmath,amssymb,amsfonts,amsthm}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{listings}
\RequirePackage{booktabs}
\RequirePackage{xcolor}
\RequirePackage[colorlinks=true]{hyperref}
\RequirePackage{textcomp}
\RequirePackage{color}
\RequirePackage{algorithm}
\if@ieee
    \RequirePackage{cite}
\else
    \RequirePackage[numbers,sort&compress]{natbib}
\fi
\RequirePackage{algorithmicx}
\RequirePackage{algpseudocode}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\RequirePackage{etoolbox}
\RequirePackage{fancyhdr}
\RequirePackage{titling}
\RequirePackage{setspace}
\RequirePackage{arithmax-logo}

% Font setup (copied from acmart)
\RequirePackage[T1]{fontenc}
\IfFileExists{libertine.sty}{\RequirePackage[tt=false, type1=true]{libertine}}{}
\IfFileExists{zi4.sty}{\RequirePackage[varqu]{zi4}}{}
\IfFileExists{newtxmath.sty}{\RequirePackage[libertine]{newtxmath}}{}

% PGF plots compatibility
\pgfplotsset{compat=1.18}

% Algorithm spacing adjustments
\AtBeginEnvironment{algorithm}{\vspace{-0.5em}}
\AfterEndEnvironment{algorithm}{\vspace{-0.5em}}

% Geometry settings based on class type
\if@ieee
    \if@onecolumn
        \geometry{
            a4paper,
            left=2.5cm,
            right=2.5cm,
            top=2.5cm,
            bottom=2.5cm
        }
        \setstretch{1.1}
    \fi
\else
    \geometry{a4paper, margin=1in}
\fi

% Header and footer setup with proper head height
\setlength{\headheight}{50pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\arithmaxheaderlogo[1.2cm]}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\small\textcolor{arithmaxgray}{Arithmax Research - Confidential}}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{arithmaxblue}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\footrule}{\hbox to\headwidth{\color{arithmaxgray}\leaders\hrule height \footrulewidth\hfill}}

% Title page formatting
\pretitle{\begin{center}\LARGE\bfseries\color{arithmaxblue}}
\posttitle{\par\end{center}\vskip 0.5em}

% Author formatting commands
\if@ieee
    \newcommand{\arithmaxauthor}[3]{%
        \IEEEauthorblockN{#1}%
        \IEEEauthorblockA{\textit{Arithmax Research} \\
        #2 \\
        Email: #3}%
    }
\else
    \newcommand{\arithmaxauthor}[3]{%
        \author{%
            \textbf{#1} \\
            \textit{Arithmax Research} \\
            #2 \\
            \texttt{#3}%
        }%
    }
\fi

% Abstract environment customization
\if@ieee
    % Keep IEEE abstract as is
\else
    \renewenvironment{abstract}{%
        \small
        \begin{center}%
            {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
        \noindent\ignorespaces
    }{%
        \endquotation
    }
\fi

% Section formatting
\if@ieee
    % Keep IEEE section formatting
\else
    \RequirePackage{titlesec}
    \titleformat{\section}{\large\bfseries\color{arithmaxblue}}{\thesection}{1em}{}
    \titleformat{\subsection}{\normalsize\bfseries\color{arithmaxblue}}{\thesubsection}{1em}{}
    \titleformat{\subsubsection}{\normalsize\bfseries\color{arithmaxgray}}{\thesubsubsection}{1em}{}
\fi

% BibTeX definition (for IEEE compatibility)
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

% Hyperlink colors
\hypersetup{
    colorlinks=true,
    linkcolor=arithmaxblue,
    citecolor=arithmaxblue,
    urlcolor=arithmaxblue,
    filecolor=arithmaxblue
}

% Listing settings
\lstset{
    backgroundcolor=\color{arithmaxlightgray},
    commentstyle=\color{arithmaxgray},
    keywordstyle=\color{arithmaxblue},
    numberstyle=\tiny\color{arithmaxgray},
    stringstyle=\color{arithmaxblue},
    basicstyle=\footnotesize\ttfamily,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    frame=single,
    frameround=tttt,
    rulecolor=\color{arithmaxgray}
}

% Research paper environments
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{note}[theorem]{Note}

% Custom commands for common mathematical notation
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\text{Var}}
\newcommand{\Cov}{\text{Cov}}
\newcommand{\Prob}{\mathbb{P}}

% Financial mathematics notation
\newcommand{\price}[1]{P_{#1}}
\newcommand{\return}[1]{r_{#1}}
\newcommand{\vol}[1]{\sigma_{#1}}
\newcommand{\drift}{{\mu}}

% Algorithm formatting for non-IEEE classes
\if@ieee
    % Keep IEEE algorithm formatting
\else
    \makeatletter
    \renewenvironment{algorithm}[1][h]
      {\begin{figure}[#1]
       \centering
       \begin{minipage}{0.9\textwidth}
       \hrule
       \vspace{0.3em}}
      {\vspace{0.3em}
       \hrule
       \end{minipage}
       \end{figure}}
    \makeatother
\fi

\endinput