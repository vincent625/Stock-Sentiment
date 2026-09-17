# Input data contract

The analysis intentionally uses a vendor-neutral input contract. Convert LSEG, RavenPack, CRSP, Nasdaq, or another feed to these columns before running the study.

## News events

Required:

| column | type | meaning |
|---|---|---|
| `story_id` | string | stable story/article identifier |
| `first_timestamp_utc` | UTC timestamp | first time the information became available |
| `ticker` | string | affected security |
| `sentiment` | float [-1, 1] | directional sentiment |

Optional but recommended:

| column | default | meaning |
|---|---:|---|
| `issuer_id` | ticker | issuer identity; use the same id for GOOG/GOOGL |
| `relevance` | 1.0 | entity relevance [0,1] |
| `novelty` | 1.0 | information novelty [0,1] |
| `source_quality` | 1.0 | source-quality weight [0,1] |
| `event_type` | unknown | earnings, guidance, M&A, product, litigation, etc. |
| `source` | unknown | Reuters, company release, Bloomberg, etc. |

Do not replace `first_timestamp_utc` with an ingestion time. The timestamp must represent when the information was first observable to the market.

## Price bars

Required:

| column | type | meaning |
|---|---|---|
| `ticker` | string | security symbol; include benchmark ticker such as QQQ |
| `timestamp_utc` | UTC timestamp | bar timestamp |
| `close` | positive float | corporate-action-consistent close for the bar |

For reaction-speed work, use 1-minute or finer bars. For daily drift, use research-grade adjusted daily prices/returns.

## Story updates and syndication

Deduplicate repeated copies before analysis. A story update should not create multiple independent observations unless the update contains genuinely new information and has its own novelty score.
