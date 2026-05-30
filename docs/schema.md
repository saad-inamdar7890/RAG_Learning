# Document + Chunk Schema

## Document
- doc_id: stable unique id (e.g., INS-0001)
- source_url
- title
- source_org
- section_heading (optional)
- page_number
- raw_text

## Chunk
- chunk_id
- doc_id
- chunk_index
- text
- start_char
- end_char
- page_number

## Notes
- Every chunk must map back to a document and page for citations.
