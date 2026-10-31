"""Extract archived source-tweet counts; this does not fit a prediction model."""
from pathlib import Path, PurePosixPath
import argparse, json, tarfile
import pandas as pd
TARGETS = ('retweet_count', 'favorite_count')
def event_from_directory(directory: str) -> str:
    suffix = "-all-rnr-threads"
    return directory[: -len(suffix)] if directory.endswith(suffix) else directory

def extract_engagement(archive_path: Path, wanted_ids: set[str]) -> pd.DataFrame:
    """Extract source-tweet counters and author reach from the PHEME archive."""
    rows: list[dict] = []
    invalid = 0
    with tarfile.open(archive_path, mode="r:*") as bundle:
        for member in bundle.getmembers():
            parts = PurePosixPath(member.name).parts
            if (
                len(parts) != 6
                or parts[4] not in {"source-tweet", "source-tweets"}
                or not parts[-1].endswith(".json")
            ):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            try:
                tweet = json.load(extracted)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid += 1
                continue
            tweet_id = str(tweet.get("id_str") or tweet.get("id") or parts[3])
            if tweet_id not in wanted_ids:
                continue
            user = tweet.get("user") or {}
            text = str(tweet.get("full_text") or tweet.get("text") or "")
            rows.append(
                {
                    "id": tweet_id,
                    "event_from_archive": event_from_directory(parts[1]),
                    "retweet_count": tweet.get("retweet_count"),
                    "favorite_count": tweet.get("favorite_count"),
                    "author_id": str(user.get("id_str") or user.get("id") or ""),
                    "followers_count": user.get("followers_count"),
                    "friends_count": user.get("friends_count"),
                    "statuses_count": user.get("statuses_count"),
                    "author_favourites_count": user.get("favourites_count"),
                    "listed_count": user.get("listed_count"),
                    "created_at": tweet.get("created_at"),
                    "text_length": len(text),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("id", keep="first")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate tweet IDs remain after PHEME extraction")
    missing = wanted_ids - set(frame["id"])
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} prepared PHEME IDs")
    for column in [*TARGETS, "followers_count", "text_length"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(TARGETS)].isna().any().any():
        raise ValueError("Missing engagement counters in extracted source tweets")
    print(
        f"Extracted {len(frame):,} aligned PHEME source tweets "
        f"({invalid} invalid JSON files skipped globally)."
    )
    return frame
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('data/pheme/PHEME_veracity.tar.bz2'))
    args=parser.parse_args()
    ids=set(pd.read_csv('data/pheme/pheme_ready.csv',dtype={'id':str}).id)
    out=Path('runs/pheme_embedding_comparison/engagement');out.mkdir(parents=True,exist_ok=True)
    extract_engagement(args.archive,ids).to_csv(out/'pheme_engagement_extracted.csv',index=False)
