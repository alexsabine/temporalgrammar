#!/bin/sh
# Builds ../../fireflies.html from the parts here. Run from this folder.
set -e
{ cat head.html; printf '\n<script>\n'; cat model.js; printf '</script>\n<script>\n'; cat view.js; printf '</script>\n</body>\n</html>\n'; } > ../../fireflies.html
echo "built ../../fireflies.html ($(wc -c < ../../fireflies.html) bytes)"
