# GitHub Action

Scan every pull request, post the summary as a comment, and keep the full report as an
artifact. Nothing is sent anywhere except the comment on your own repository.

```yaml
# .github/workflows/llm-cost.yml
name: LLM cost
on:
  pull_request:
  schedule:
    - cron: "17 6 * * 1"     # weekly, with usage data

permissions:
  contents: read
  pull-requests: write          # only needed for the PR comment

jobs:
  trainlab:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: trainlab
        uses: trainlab-ai/trainlab-optimize@v0      # pin to a commit SHA in production
        with:
          exclude: examples,docs
          fail-on: none                            # or: high
          # Weekly run with real spend (admin keys have read-only usage access):
          openai-admin-key: ${{ github.event_name == 'schedule' && secrets.OPENAI_ADMIN_KEY || '' }}
      - uses: actions/upload-artifact@v4
        with:
          name: trainlab-report
          path: ${{ steps.trainlab.outputs.report-dir }}
```

| Input | Default | Meaning |
|---|---|---|
| `path` | `.` | Folder(s) to scan, space-separated |
| `exclude` | | Comma-separated globs to skip |
| `usage-csv` | | Billing export(s) to import |
| `openai-admin-key`, `anthropic-admin-key` | | Usage API keys (use secrets) |
| `fail-on` | `none` | Fail when findings at or above `high` / `medium` / `low` exist |
| `comment` | `true` | Post or update a PR comment |

Outputs: `report-dir`, `findings`, `monthly-spend`.
