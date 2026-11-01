#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f .env ]]; then cp .env.example .env; fi
if grep -q '^WATERMARK_RESEARCH_KEY=replace_with_a_private_integer$' .env; then
  key="$(python3 -c 'import secrets; print(secrets.randbits(63))')"
  sed -i "s/^WATERMARK_RESEARCH_KEY=.*/WATERMARK_RESEARCH_KEY=${key}/" .env
fi
chmod 600 .env
