#!/bin/sh
# git askpass helper: credentials come from the environment of the single
# push subprocess, so the token never touches disk, argv, or git config.
case "$1" in
  *sername*) echo "$GIT_USERNAME" ;;
  *) echo "$GIT_PASSWORD" ;;
esac
