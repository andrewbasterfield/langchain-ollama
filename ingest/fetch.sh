#!/bin/sh

curl -s https://tldp.org/HOWTO/text/ | perl -lne 'm/^<li><p><b><a href="(.*)">/ && print $1' | while read doc; do curl -s -o $doc.txt https://tldp.org/HOWTO/text/$doc; done
