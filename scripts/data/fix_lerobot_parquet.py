import sys
from pathlib import Path
import pyarrow.parquet as pq

for path in Path(sys.argv[1]).rglob("*.parquet"):
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    meta.pop(b"huggingface", None)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(meta or None), tmp)
    print("fixing ", path)
    tmp.replace(path)
