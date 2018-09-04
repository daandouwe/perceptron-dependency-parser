#!/usr/bin/env bash

echo "Downloader for Universal Dependencies."
echo "Default download is English."
read -r -p "Choose a second language from (cs, de, es, fr, hi, nl): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"

if [[ ! -d ud ]]; then
    mkdir ud
fi

if [[ ! -d ud/UD_English-EWT ]]; then  # makes it easier to reuse script for multiple language download
    mkdir ud/UD_English-EWT
    wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu \
        -P ud/UD_English-EWT
    wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu \
        -P ud/UD_English-EWT
    wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu \
        -P ud/UD_English-EWT
fi

if [[ $LANG == 'nl' ]]; then
    DIR=UD_Dutch-Alpino
    NAME=nl_alpino
fi
if [[ $LANG == 'fr' ]]; then
    DIR=UD_French-FTB
    NAME=fr_ftb
fi
if [[ $LANG == 'de' ]]; then
    DIR=UD_German-GSD
    NAME=de_gsd
fi
if [[ $LANG == 'hi' ]]; then
    DIR=UD_Hindi-HDTB
    NAME=hi_hdtb
fi
if [[ $LANG == 'es' ]]; then
    DIR=UD_Spanish-GSD
    NAME=es_gsd
fi
if [[ $LANG == 'cs' ]]; then
    DIR=UD_Czech-PDT
    NAME=cs_pdt
    # The Czech train data is irregularly named.
    wget https://raw.githubusercontent.com/UniversalDependencies/$DIR/master/$NAME-ud-train-l.conllu
    mv $NAME-ud-train-l.conllu $NAME-ud-train.conllu
fi

# And many more languages...

mkdir ud/$DIR
wget https://raw.githubusercontent.com/UniversalDependencies/$DIR/master/$NAME-ud-train.conllu \
    -P ud/$DIR
wget https://raw.githubusercontent.com/UniversalDependencies/$DIR/master/$NAME-ud-dev.conllu \
    -P ud/$DIR
wget https://raw.githubusercontent.com/UniversalDependencies/$DIR/master/$NAME-ud-test.conllu \
    -P ud/$DIR
