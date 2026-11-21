#!/bin/sh
# Assembles ../../whale.html from the parts in this folder.
cd "$(dirname "$0")"
{ cat head.html; echo '<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>'; echo '<script>'; cat model.js; echo '</script>'; echo '<script>'; cat app.js; echo '</script>'; echo '</body>'; echo '</html>'; } > ../../whale.html
wc -c ../../whale.html
