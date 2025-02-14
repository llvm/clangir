#!/bin/bash

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

REBASE_BRANCH=rebased-clangir-onto-llvm-upstream
TARGET_BRANCH=$1

log() { printf "%b[%s]%b %s\n" "$1" "$2" "$NC" "$3"; }
log_info() { log "$GREEN" "INFO" "$1"; }
log_warn() { log "$YELLOW" "WARN" "$1"; }
log_error() { log "$RED" "ERROR" "$1" >&2; }

error_exit() {
    log_error "$1"
    exit 1
}

is_rebasing() {
    git rev-parse --git-path rebase-merge >/dev/null 2>&1 ||
        git rev-parse --git-path rebase-apply >/dev/null 2>&1
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    error_exit "Not in a Git repository."

git remote get-url upstream >/dev/null 2>&1 ||
    error_exit "Upstream remote not found."

log_info "Fetching latest changes from upstream..."

git fetch upstream main ||
    error_exit "Failed to fetch from upstream."

REBASING_CIR_BRANCH=$(git branch --show-current)

if [ -z "$COMMON_ANCESTOR" ]; then
    COMMON_ANCESTOR=$(git merge-base upstream/main "$REBASING_CIR_BRANCH") ||
        error_exit "Could not find common ancestor."
    log_info "Common ancestor commit: $COMMON_ANCESTOR"
fi

if [ "$REBASING_CIR_BRANCH" != "$REBASE_BRANCH" ]; then
    if git rev-parse --verify "$REBASE_BRANCH" >/dev/null 2>&1; then
        git branch -D "$REBASE_BRANCH" >/dev/null 2>&1 ||
            log_warn "Failed to delete existing branch $REBASE_BRANCH."
    fi

    git switch -c "$REBASE_BRANCH" "$COMMON_ANCESTOR" ||
        error_exit "Failed to create branch $REBASE_BRANCH."
fi

#
# Rebase upstream changes
#
log_info "Processing upstream commits..."
git rebase upstream/main ||
    error_exit "Failed to rebase."

#
# Reverse upstream CIR commits
#
log_info "Reverting upstream CIR commits..."
git log --grep="\[CIR\]" --format="%H %s" "$COMMON_ANCESTOR..upstream/main" | while read -r HASH MESSAGE; do
    log_info "Reverting: $MESSAGE"

    if ! git revert --no-edit "$HASH"; then
        error_exit "Failed to revert commit $HASH"
    fi
done

#
# Rebase CIR commits
#
log_info "Rebasing CIR incubator commits..."

if [ -z "$TARGET_BRANCH" ]; then
    log_error "Target branch not specified."
    exit 1
fi

if git rev-parse --verify "$TARGET_BRANCH" >/dev/null 2>&1; then
    git branch -D "$TARGET_BRANCH" >/dev/null 2>&1 ||
        error_exit "Failed to delete existing branch $TARGET_BRANCH."
fi

git switch "$REBASING_CIR_BRANCH" ||
    error_exit "Failed to switch to $REBASING_CIR_BRANCH."
git checkout -b "$TARGET_BRANCH" ||
    error_exit "Failed to checkout $TARGET_BRANCH."
git rebase "$REBASE_BRANCH" "$TARGET_BRANCH" -X theirs ||
    error_exit "Failed to rebase."

log_info "Rebase completed successfully!"
